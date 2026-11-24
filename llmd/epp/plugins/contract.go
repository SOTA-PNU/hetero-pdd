package plugins

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"strings"

	eppErr "github.com/llm-d/llm-d-router/pkg/common/error"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	dl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	rc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	rh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	s "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type contract struct {
	named
	Reason      string `json:"reason"`
	ServedModel string `json:"servedModel"`
}

func newContract(n named, d *json.Decoder) (plugin.Plugin, error) {
	p := &contract{named: n}
	return p, decode(d, p)
}

type requestGuard struct{ named }

func (*requestGuard) RequestHeader(_ context.Context, r *s.InferenceRequest) error {
	for k := range r.Headers {
		if strings.HasPrefix(strings.ToLower(k), "x-hetero-") {
			delete(r.Headers, k)
		}
	}
	for _, h := range []string{routing.PrefillEndpointHeader, routing.EncoderEndpointsHeader, routing.KVCacheSourceHeader} {
		routing.DeleteRoutingHeader(r.Headers, h)
	}
	return nil
}
func (*requestGuard) Admit(_ context.Context, r *s.InferenceRequest, _ []s.Endpoint) error {
	if r.Body != nil {
		if m, ok := r.Body.Payload.(rh.PayloadMap); ok {
			if _, present := m["kv_transfer_params"]; present {
				return eppErr.Error{Code: eppErr.BadRequest, Msg: "kv_transfer_params is reserved for the PD sidecar"}
			}
		} else if len(r.Body.RawBody) > 0 {
			var m map[string]json.RawMessage
			if err := json.Unmarshal(r.Body.RawBody, &m); err == nil && m["kv_transfer_params"] != nil {
				return eppErr.Error{Code: eppErr.BadRequest, Msg: "kv_transfer_params is reserved for the PD sidecar"}
			}
		}
	}
	return nil
}
func (c *contract) PreRequest(ctx context.Context, r *s.InferenceRequest, res *s.SchedulingResult) error {
	if c.ServedModel != "" && r.TargetModel != c.ServedModel {
		return eppErr.Error{Code: eppErr.BadRequest, Msg: "model outside this deployment"}
	}
	d := res.PrimaryEndpoint()
	if d == nil || endpointID(d) == "" {
		return fmt.Errorf("selected endpoint has no instance ID")
	}
	h := r.Headers
	h["x-hetero-policy"] = c.name.Name
	if c.Reason != "" {
		h["x-hetero-reason"] = c.Reason
	}
	h["x-hetero-target"] = endpointID(d)
	h["x-hetero-path"] = "full"
	routing.DeleteRoutingHeader(h, routing.PrefillEndpointHeader)
	p := res.ProfileResults["prefill"].FirstEndpoint()
	if p != nil {
		if !(pairConfig{}).compatible(p, d) {
			return fmt.Errorf("incompatible selected pair")
		}
		m := p.GetMetadata()
		engineID := label(p, "engine-id")
		switch source := label(p, "engine-id-source"); source {
		case "":
		case "pod-name":
			// Pod discovery stores pod.Name here. Metadata.ID.Name includes a
			// rank suffix and is not the engine ID from the Downward API.
			engineID = m.Name
		default:
			return fmt.Errorf("unsupported producer engine ID source %q", source)
		}
		port := label(p, "engine-port")
		if port == "" {
			port = m.Port
		}
		routing.SetRoutingHeader(h, routing.PrefillEndpointHeader, net.JoinHostPort(m.Address, port))
		h["x-hetero-path"] = "pdd"
		h["x-hetero-prefill"] = endpointID(p)
		for key, v := range map[string]string{"engine": engineID, "host": m.Address, "port": label(p, "kv-port"), "tp": label(p, "tp"), "block-size": label(p, "block-size"), "num-blocks": label(p, "num-blocks")} {
			if v == "" {
				return fmt.Errorf("missing producer %s for %s", key, endpointID(p))
			}
			h["x-hetero-expected-"+key] = v
		}
	}
	log.FromContext(ctx).Info("hetero_route", "request_id", h["x-request-id"], "epp_request_id", r.RequestID, "model", r.TargetModel, "path", h["x-hetero-path"], "target", h["x-hetero-target"], "prefill", h["x-hetero-prefill"], "policy", c.name.Name)
	return nil
}
func (*contract) ResponseBody(ctx context.Context, r *s.InferenceRequest, resp *rc.Response, _ *dl.EndpointMetadata) {
	if resp.EndOfStream {
		log.FromContext(ctx).Info("hetero_route_finished", "request_id", r.Headers["x-request-id"], "epp_request_id", r.RequestID, "cause", resp.TerminationCause, "usage", resp.Usage)
	}
}

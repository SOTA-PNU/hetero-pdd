# Repository mirror

`SOTA-PNU/hetero-pdd` is the development repository. The **Mirror to pnu-yk**
workflow copies all branches, tags, and their Git history to
`pnu-yk/hetero-pdd` in one direction.

The destination is a dedicated mirror: source force-pushes overwrite its refs,
and branches or tags absent from the source are deleted from the destination.
Do not develop directly in the destination. Only `refs/heads/*` and
`refs/tags/*` are synchronized; GitHub-managed pull request refs are excluded.
Issues, pull requests, repository settings, secrets, and release attachments
are not copied. Git LFS objects, if introduced, need a separate transfer step.

## Authentication and initial setup

1. Add the mirror's SSH public key under the destination repository's
   **Settings > Deploy keys** with **Allow write access** enabled.
2. Save the corresponding private key as the source repository's Actions
   secret named `MIRROR_SSH_PRIVATE_KEY`. Never commit the private key.
3. Merge `.github/workflows/mirror.yml` into the source default branch.
4. Open **Actions > Mirror to pnu-yk > Run workflow** on `main` and verify
   that the run succeeds and destination branch/tag SHAs match the source.

The source is public, so fetching it needs no credential. The deploy key grants
write access only to the destination. GitHub's published Ed25519 host key is
pinned in the workflow. Destination rulesets must permit the ref updates the
mirror performs; failures are reported rather than bypassed.

## Operation

- Pushes trigger synchronization when the pushed ref contains this workflow.
- Ref creation and deletion events also trigger synchronization.
- An hourly schedule at minute 23 reconciles updates on existing branches
  that do not yet contain the workflow, and events missed by GitHub Actions.
  Scheduled runs can be delayed by GitHub; this is not an hourly SLA.
- Manual dispatch provides an immediate full synchronization from `main`.
- Runs are serialized and clone current source refs at execution time.
  Atomic pushes prevent a partially updated mirror if a ref is rejected.
- The repository guard prevents this mirroring job from running in the
  destination. Other copied workflows retain their own triggers; manage
  destination Actions settings separately if those workflows are unwanted.

GitHub may disable scheduled workflows in public repositories after 60 days
of inactivity. Re-enable the workflow or use manual dispatch when resuming.
Check failed Actions runs for key revocation, destination ruleset changes, or
host key changes. To rotate the key, register the new destination deploy key,
update the source secret, verify a successful run, then remove the old key.

References: [Deploy keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys)
and [workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).

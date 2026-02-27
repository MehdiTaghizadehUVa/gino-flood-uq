# Let the agent push to GitHub from UVA Rivanna (SSH)

On Rivanna, git cannot prompt for passwords, so pushes fail. Use one of the options below so that **non-interactive** pushes (e.g. by the Cursor agent) work.

---

## Option 1: SSH key (recommended)

Git uses SSH to talk to GitHub without any prompt. You do a one-time setup on Rivanna and in GitHub.

### 1.1 Generate an SSH key on Rivanna (if you don’t already have one)

On Rivanna, in a terminal:

```bash
# Use your @virginia.edu or preferred email
ssh-keygen -t ed25519 -C "your_email@virginia.edu" -f ~/.ssh/id_ed25519_github -N ""
```

`-N ""` means no passphrase, so the agent can push without interaction. If you prefer a passphrase, you’ll need to add the key to `ssh-agent` in your session before the agent runs.

### 1.2 Add the public key to GitHub

1. Show the public key:
   ```bash
   cat ~/.ssh/id_ed25519_github.pub
   ```
2. Copy the full line.
3. On GitHub: **Settings** → **SSH and GPG keys** → **New SSH key** → paste and save.

### 1.3 Tell SSH to use this key for GitHub

Create or edit `~/.ssh/config` on Rivanna:

```
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
```

### 1.4 Use the SSH remote in this repo

From the project root:

```bash
cd /path/to/neuraloperator_no_physics
git remote set-url github git@github.com:MehdiTaghizadehUVa/gino-flood-uq.git
git push -u github main
```

After this, any `git push github main` (including from the agent) will use SSH and succeed without prompts.

---

## Option 2: Personal Access Token (HTTPS)

If you prefer HTTPS, GitHub can use a **Personal Access Token (PAT)** instead of a password. Store it once so future pushes are non-interactive.

### 2.1 Create a PAT on GitHub

1. **Settings** → **Developer settings** → **Personal access tokens** → **Tokens (classic)**.
2. **Generate new token (classic)**.
3. Name it (e.g. `Rivanna gino-flood-uq`), set an expiration, and enable scope **`repo`**.
4. Generate and **copy the token** (you won’t see it again).

### 2.2 Store the token on Rivanna (one-time)

On Rivanna, in an **interactive** session (so git can ask once):

```bash
cd /path/to/neuraloperator_no_physics

# Store credentials in plain text after one successful login (use PAT as password)
git config --local credential.helper store
git push -u github main
# Username: MehdiTaghizadehUVa
# Password: <paste your PAT>
```

After this, credentials are in `~/.git-credentials`. The agent can run `git push github main` without prompts.

**Security:** Restrict PAT scope to `repo` and rotate it if needed. Prefer Option 1 (SSH) if possible.

---

## Quick check

From the project directory on Rivanna:

```bash
# If using SSH (Option 1):
git remote get-url github
# Should show: git@github.com:MehdiTaghizadehUVa/gino-flood-uq.git

# Test push (no prompt should appear)
git push github main
```

If that works, the agent can push by itself from Rivanna.

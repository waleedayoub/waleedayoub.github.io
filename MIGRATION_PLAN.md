# Theme Migration Plan: gruvhugo → drishtikon

**Goal**: Migrate to drishtikon theme - minimal two-column layout with bio on left, posts on right.

**Theme Repo**: https://github.com/kishenarayan/drishtikon

---

## Step 1: Install Theme

```bash
git submodule add https://github.com/kishenarayan/drishtikon.git themes/drishtikon
```

---

## Step 2: Rename Content Directory

Drishtikon expects `posts/` not `post/`:

```bash
mv content/post content/posts
```

---

## Step 3: Update hugo.toml

Replace current config with:

```toml
baseURL = "https://waleedayoub.com"
languageCode = "en-us"
title = "Waleed Ayoub"
theme = "drishtikon"

[params]
  profile_photo = "/images/profile.jpg"
  description = "Your bio text here - can include [links](url) and *formatting*"
  updates = "Latest posts and thoughts on data, ML, and building things."
  github = "https://github.com/waleedayoub"
  email = "your@email.com"
  linkedin = "https://linkedin.com/in/waleedayoub"
  math = true

[markup]
  [markup.goldmark]
    [markup.goldmark.extensions]
      table = true
      footnote = true
      strikethrough = true
      taskList = true
    [markup.goldmark.renderer]
      unsafe = true
  [markup.highlight]
    style = "monokai"
```

---

## Step 4: Add Profile Photo

```bash
mkdir -p static/images
cp /path/to/your/photo.jpg static/images/profile.jpg
```

---

## Step 5: Keep Mermaid Support

Preserve this file (don't delete it):
- `layouts/_default/_markup/render-codeblock-mermaid.html`

---

## Step 6: Clean Up Old Theme Files

```bash
rm layouts/index.html
rm layouts/_default/baseof.html
rm layouts/_default/single.html
rm layouts/_default/list.html
rm -rf layouts/partials/
rm -rf assets/sass/
rm static/css/custom-style.css
```

---

## Step 7: Test Locally

```bash
hugo server
```

Verify:
- [ ] Two-column layout displays
- [ ] Profile photo and bio on left
- [ ] Posts list with dates on right
- [ ] Individual posts render correctly
- [ ] KaTeX math works
- [ ] Mermaid diagrams work
- [ ] Social links in footer
- [ ] Mobile responsive

---

## Step 8: Deploy

Push changes to GitHub. The existing GitHub Actions workflow should handle deployment.

---

## Files Summary

| Action | Files |
|--------|-------|
| Modify | `hugo.toml` |
| Rename | `content/post/` → `content/posts/` |
| Add | `static/images/profile.jpg`, `themes/drishtikon` (submodule) |
| Remove | `layouts/index.html`, `layouts/_default/baseof.html`, `layouts/_default/single.html`, `layouts/_default/list.html`, `layouts/partials/*`, `assets/sass/`, `static/css/custom-style.css` |
| Keep | `layouts/_default/_markup/render-codeblock-mermaid.html` |

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Local development with live reload
hugo server

# Production build
hugo build

# Create new blog post
hugo new post/my-post-title.md
```

CI/CD is handled by GitHub Actions (`.github/workflows/hugo.yaml`) which builds with Hugo 0.145.0 extended and deploys to GitHub Pages.

## Architecture

This is a **Hugo static site** (not Jekyll) for waleedayoub.com.

### Content Structure
- `content/post/` - Blog posts in Markdown with YAML or TOML frontmatter
- `content/about.md` - About page
- `data/` - YAML data files for homepage and gallery

### Theme & Styling
- **Active theme**: `gruvhugo` (Gruvbox color scheme, git submodule)
- `assets/sass/main.scss` - Main SCSS compiled by Hugo
- `static/css/custom-style.css` - Custom overrides (Mermaid, code blocks)
- Dark/light mode toggle persists to localStorage

### Template Hierarchy
- `layouts/_default/baseof.html` - Base wrapper
- `layouts/_default/single.html` - Individual post template
- `layouts/_default/list.html` - Archive/list template
- `layouts/partials/` - Reusable components (head, header, footer, menu)
- `layouts/_default/_markup/render-codeblock-mermaid.html` - Mermaid diagram support

### Configuration
`hugo.toml` defines:
- Goldmark markdown with tables, footnotes, task lists, typographer
- Monokai syntax highlighting
- KaTeX math support (CDN)
- Taxonomies: tags, categories, series

### External Dependencies (CDN)
- KaTeX for LaTeX math rendering
- Mermaid.js for diagrams
- Google Fonts (Quicksand)

## Post Frontmatter

```yaml
---
title: "Post Title"
author: waleed
date: 2024-01-15
description: "Brief description"
tags: ["tag1", "tag2"]
series: ["series-name"]
draft: false
---
```

## Key Files
- `hugo.toml` - Main configuration
- `CNAME` - Custom domain (waleedayoub.com)
- `.gitmodules` - Theme submodule references

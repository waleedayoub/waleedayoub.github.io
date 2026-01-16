---
title: "AI for Product Managers: Building Intuition Through Hands-On Examples"
author: waleed
date: 2026-01-15
description: "A 60-minute workshop on using AI effectively as a PM — from mental models to a complete workflow demo taking an idea from research to Jira tickets."
tags: ["ai", "product-management", "workflows", "claude", "chatgpt"]
draft: false
---

I recently gave a workshop to a group of product managers on building AI intuition. Not the "AI will change everything" hand-waving, but practical patterns for incorporating AI into daily PM work.

The core insight:
- The best way to build the intuition needed to build AI into your products is by becoming a power user in your own domain
- We can simplify the user experience into two modes: "chat mode" and "task mode"

The workshop covers:
- **Two mental models**: Thinking Partner (brainstorming, research, synthesis) vs. Task Automator (reading from SharePoint, writing to Confluence, creating Jira tickets)
- **The CRAFT framework**: Context, Role, Ask, Format, Tone — a diagnostic tool for when prompts aren't working
- **Skills**: How to encode your expertise into reusable AI instructions that scale across your team
- **MCPs and integrations**: How to plug your AI partner into all the tools you use to do your work
- **A complete demo**: Taking an idea from research → interviews → PRD → user stories → Jira tickets

## The Presentation

Use arrow keys or swipe to navigate. Press `O` for slide overview, `F` for fullscreen.

<div style="position: relative; width: 100%; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
  <iframe
    src="/presentations/ai-for-pms.html"
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
    allowfullscreen>
  </iframe>
</div>

---

## Try It Yourself: The Demo Workflow

The live demo in the presentation walks through a complete PM workflow. Here's how to reproduce it yourself.

### What You'll Need

| Tool                  | Purpose                                                  | Required?           |
| --------------------- | -------------------------------------------------------- | ------------------- |
| ChatGPT Plus ($20/mo) | Deep Research for async market research                  | Optional but useful |
| Claude Pro ($20/mo)   | Projects for document context, Desktop for orchestration | Yes                 |
| Claude Desktop        | Local app with MCP support                               | Yes                 |
| MCP Servers           | Confluence, Slack, Jira integrations                     | For full automation |

### The 10-Step Workflow

#### Step 1: Kick Off Deep Research

Start with an idea and let AI do async research across dozens of sources.

**Tool**: ChatGPT with Deep Research enabled

**Prompt**:
```
Research the current state of AI-powered demand planning tools for retail.
I want to understand:
- What solutions exist today (vendors, approaches)
- Common pain points demand planners face
- How AI is being applied to forecasting and inventory decisions
- What gaps exist in current offerings

Focus on practical implementation, not theoretical capabilities.
```

ChatGPT will spend 5-15 minutes researching and return a structured report with citations. Save this output — you'll feed it into the next steps.

#### Step 2: Synthesize Customer Interviews

Upload interview transcripts and extract themes, contradictions, and key quotes.

**Tool**: Claude.ai Projects (create a new project and upload your transcripts)

**Prompt**:
```
Analyze these interview transcripts from demand planners. I need:

1. Top 5 pain points (ranked by frequency and intensity)
2. Contradictions between interviewees (where people disagree)
3. Surprising insights (things I wouldn't have predicted)
4. Representative quotes for each major theme

Be specific. Cite which interviewee said what.
```

#### Step 3: Generate a PRD

Draft structured requirements from your research and interviews.

**Tool**: Claude Desktop (with your research and interview synthesis in context)

**Prompt**:
```
Based on the deep research and interview transcripts in this project, write a PRD
for an "AI-powered demand planning assistant."

The product should help demand planners:
- Get recommendations on which forecast items need attention
- Understand WHY the AI is flagging something (with evidence)
- Make faster, more confident decisions during forecast review cycles

Structure the PRD as:
1. Problem Statement — What's broken today? (cite specific pain points from interviews)
2. Target Users — Who is this for? What do they care about?
3. Goals & Success Metrics — How do we know this worked?
4. Core Capabilities — What does v1 do? (be specific, not vague)
5. Out of Scope — What are we NOT building in v1?
6. Key Risks — What could go wrong? How do we mitigate?

Be opinionated. If the research points in a clear direction, say so.
Flag any gaps where we need more information.
```

#### Step 4: (Optional) Generate a Visual Prototype

Turn your PRD into a working UI prototype.

**Tool**: Lovable, v0, or similar

**Prompt**:
```
Create a dashboard for demand planners that shows:
- A prioritized list of forecast items needing attention
- For each item: the AI's recommendation, confidence level, and supporting evidence
- Ability to accept/reject/modify recommendations
- A weekly summary view

Use a clean, professional design. The user is a busy demand planner
who needs to make decisions quickly.
```

#### Step 5: (Optional) Review the PRD

Get a critical second opinion on your draft.

**Tool**: Claude Desktop

**Prompt**:
```
Review this PRD as a senior product leader. Be critical. Identify:
- Gaps in the requirements
- Assumptions that need validation
- Risks I haven't considered
- Areas that are too vague for engineering to implement

Don't be nice. I need the hard feedback now, not after we've started building.
```

#### Step 6: Publish to Confluence

Push the PRD to your team's documentation system.

**Tool**: Claude Desktop with Confluence MCP

**Prompt**:
```
Publish this PRD to Confluence:
- Space: [YOUR_SPACE_KEY]
- Parent page: "Product Requirements"
- Title: "AI Demand Planning Assistant - PRD"

Format it nicely with proper headings and a table of contents.
```

#### Step 7: Convert PRD to User Stories

Break down requirements into engineering-ready work items.

**Tool**: Claude Desktop

**Prompt**:
```
Convert this PRD into user stories for engineering.

For each story:
- Format: "As a [user], I want [capability] so that [outcome]"
- Include 3-5 acceptance criteria per story
- Note any dependencies between stories
- Flag stories that need design input before dev can start

Group the stories into epics that could be delivered incrementally.
Suggest a logical build order (what ships first vs. later).

Keep stories small enough to complete in 1-2 sprints.
If something is too big, break it down.
```

#### Step 8: Publish Stories for Review

Share the user stories with stakeholders for feedback.

**Tool**: Claude Desktop with Confluence + Slack MCPs

**Prompt**:
```
1. Publish these user stories to Confluence under the PRD page
2. Post a message to #product-reviews in Slack:
   "New user stories ready for review: AI Demand Planning Assistant.
   Please add comments directly in Confluence. Review deadline: Friday EOD."
```

#### Step 9: Triage Stakeholder Feedback

Process comments and formulate responses.

**Tool**: Claude Desktop with Confluence MCP

**Prompt**:
```
Read the comments on the AI Demand Planning user stories page. For each comment:
- Categorize as: clarification needed, valid concern, out of scope, or already addressed
- Draft a response
- Flag any that require PRD changes

Show me your analysis before posting any responses.
```

#### Step 10: Create Jira Tickets

Push approved stories to your project management tool.

**Tool**: Claude Desktop with Jira MCP

**Prompt**:
```
Create Jira stories for each approved user story:
- Project: [YOUR_PROJECT_KEY]
- Epic: "AI Demand Planning Assistant"
- Include acceptance criteria in the description
- Set story points based on complexity (use 1, 2, 3, 5, 8)
- Link back to the Confluence PRD

Show me what you're about to create before submitting.
```

---

## Key Takeaways

1. **Two modes**: Use AI as a Thinking Partner (research, synthesis, ideation) and as a Task Automator (read/write across tools). Most people only use the first.

2. **CRAFT framework**: When prompts aren't working, check Context, Role, Ask, Format, and Tone.

3. **Skills scale expertise**: Write your standards once, AI applies them forever. New team members get your expertise instantly.

4. **Always review before publish**: AI drafts, you decide. The human-in-the-loop isn't a limitation — it's where your judgment adds value.

5. **Start with one workflow**: Pick one thing you do repeatedly, try AI-assisted, iterate. Expand from there.

---

Questions or want to try this workflow together? Reach out — happy to help you get set up.

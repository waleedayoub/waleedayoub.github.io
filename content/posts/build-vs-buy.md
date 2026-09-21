---
title: "Build vs. Buy: How the Easy Parts Got Easier and the Hard Parts Got Harder"
author: waleed
date: 2026-09-21
description: "A talk on AI labor and the build-vs-buy decision — why writing code got cheap while owning it didn't, and what you're actually signing up to own when you build AI-native systems in-house."
tags: ["ai", "agents", "build-vs-buy", "engineering-leadership"]
draft: false
---

I gave this talk to a room full of people who were, at some point that day, going to hear someone on their team say "we can just build this ourselves now that AI writes the code." I wanted to give them a straight answer.

The talk makes four claims, in order, each backed by published research rather than assertion:

1. **Writing code got cheap. Owning what you wrote did not.** And ownership compounds with every release you ship — technical debt didn't go away when AI showed up, it just got easier to take on and no easier to pay down.
2. **Unlocking AI labor is organizational before it's technical.** The tools are available to everyone on identical terms. The organizations that can actually use them are not — and even Anthropic isn't at "level four" on their own maturity curve for the tool they build.
3. **What you'd actually own isn't a model or a chatbot.** It's two permanent production systems — a math engine and a governed agent harness — plus the research that keeps both current, funded forever.
4. **Your data and your logic are genuinely unique. That's an input, not a reason to build platforms.** Most of what makes a business different is configuration, not software — and conflating the two is a category error, not a bad judgment call.

## The Presentation

Use the arrow keys to move between slides, or click the left/right half of the screen. Press `N` to toggle speaker notes.

<div style="position: relative; width: 100%; padding-bottom: 56.25%; height: 0; overflow: hidden; border-radius: 8px; box-shadow: 0 4px 20px rgba(0,0,0,0.15);">
  <iframe
    src="/presentations/build-vs-buy.html"
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
    allowfullscreen>
  </iframe>
</div>

---

## The six questions worth asking before anyone authorizes a build

The talk closes with these. Ask them in the room where the decision actually gets made, and the decision mostly makes itself:

1. Who owns this in two years? Name the person, not the team.
2. How will we know it got better, before we ship it — not from a user telling us in month four?
3. What happens when the business changes? (It will. Is that a config change or a rebuild?)
4. Who is on call for it tonight?
5. What would three years of a specialist have actually cost? That's the number a build should be compared against — not zero.
6. Is this configuration? And if so, where do we say it? If saying it out loud needs a platform you haven't built, you're proposing to build software to hold a sentence.

The question was never whether you *can* build it. Of course you can. The question is which expert you should trust here, and what evidence would change your mind.

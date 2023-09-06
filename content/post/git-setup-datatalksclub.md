---
title: "Git set up for datatalksclub zoomcamp"
date: 2023-09-05T22:40:29-04:00
description: "How to set up your git repo for zoomcamp classes"
tags: ["datatalksclub","git"]
draft: false
---

## Requirements 
- Original course repo as read-only that gets updated throughout the duration of the course
- My own repo in my github account that is used to get updates from original course repo and for me to write my own assignments / work

## Solution
What I'm looking to do is essentially maintain a fork of the original course repository, while also adding your own work to it. Here's a step-by-step guide to help you set this up:

1. **Fork the Original Repository**:
   - Go to the GitHub page of the course repository.
   - Click the "Fork" button on the top right. This will create a copy of the repository under your GitHub account.

2. **Clone Your Forked Repository**:
   - On your forked repository's GitHub page, click the "Code" button and copy the URL.
   - In your terminal or command prompt, navigate to where you want to store the project and run:
     ```
     git clone [URL]
     ```
     Replace `[URL]` with the URL you copied.

3. **Set Up a Remote to Track the Original Repository**:
   - Navigate to your cloned directory:
     ```
     cd [repository-name]
     ```
   - Add the original repository as a remote named `upstream`:
     ```
     git remote add class [original-repo-URL]
     ```

4. **Syncing Your Fork**:
   Whenever you want to pull updates from the original course repository:
   - Fetch the changes:
     ```
     git fetch class
     ```
   - Make sure you're on your main branch (or whichever branch you want to update):
     ```
     git checkout main
     ```
   - Merge the changes from the `upstream`'s main branch:
     ```
     git merge class/main
     ```

5. **Adding Your Own Work**:
   - Create a new branch for your work to keep things organized:
     ```
     git checkout -b my-feature-branch
     ```
   - Make your changes, add them, and commit:
     ```
     git add .
     git commit -m "Description of my changes"
     ```
   - Push your changes to your forked repository:
     ```
     git push origin my-feature-branch
     ```

6. **Merging Your Work into Your Main Branch**:
   If you're satisfied with your work and want to merge it into your main branch:
   - Switch to your main branch:
     ```
     git checkout main
     ```
   - Merge your feature branch:
     ```
     git merge my-feature-branch
     ```
   - Push the merged changes to your forked repository:
     ```
     git push origin main
     ```

7. **Keeping Your Work Updated**:
   Periodically, you'll want to pull updates from the original course repository (as described in step 4) to ensure you have the latest content. Just be cautious when merging if you've made extensive changes to your fork, as there may be merge conflicts. Resolve any conflicts as needed.

Remember, the key is to treat the `upstream` as read-only. You'll never push changes to it. All your work will be done in your forked repository (`origin`). This setup allows you to both keep up with updates from the course and maintain your own changes.
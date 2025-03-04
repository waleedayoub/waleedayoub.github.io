---
title: "ML Zoomcamp Week 5 Homework Using Poetry and Pyenv"
author: waleed
date: 2023-10-16T19:08:29-04:00
description: "Alternative approach to MLZoomcamp's Week 5 homework using Poetry and Pyenv instead of Pipenv, with step-by-step instructions for Python package and version management."
tags: ["datatalksclub","git","mlzoomcamp","poetry","pyenv"]
draft: true
---

In order to do week5 homework, you need to use pipenv to manage python packages and dependencies

An alternative toolset would be to use poetry with pyenv

In this case, you would split the management of python applications from python versions:

- poetry would be used to manage packages, dependencies and ensure project reproducibility

- Whereas pyenv would be used to manage python versions

Install both

In your homework directory, you can run poetry init to initialize a poetry project


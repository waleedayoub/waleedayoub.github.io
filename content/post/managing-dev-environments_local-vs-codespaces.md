---
title: "Managing Dev Environments - Local vs Codespaces"
description: "The various options to running python development environments and managing secrets"
author: waleed
tags: ["python", "github", "codespaces", "secrets"]
date: 2024-07-04T07:28:08-04:00
draft: false
---

# Managing Development Environments: Local Machine vs. GitHub Codespaces

Since I don't really do development on a regular basis, especially not for my day-to-day at work, I find that one of the most annoying parts of picking up projects is getting environments set up, making sure libraries are up-to-date if needed and managing any environment variables and secrets to 3rd party services.

So this post really is me capturing the patterns I've used and find the most useful so I can refer to them in the future (instead of fumbling through half-written notes in Obsidian).

Moreover, I generally dislike running services (e.g. Docker, Elasticsearch, etc.) locally, and prefer hosting them somewhere else. For now, that usually means I go with two options for convenience and cost reasons:

- Option 1: I run them on another machine in my house, preferably a machine called `thirteen`, which is an old MacBook Pro that has enough processing power to handle Docker containers.
- Option 2: I just use something like GitHub Codespaces.

## Table of Contents
- [Virtual Environments](#virtual-environments)
  - [Option 1: Using Homebrew](#option-1-using-homebrew)
  - [Option 2: Using GitHub Codespaces](#option-2-using-github-codespaces)
- [Environment Variables](#environment-variables)
  - [Option 1: MacOS](#option-1-macos)
  - [Option 2: GitHub Codespaces](#option-2-github-codespaces)
- [References](#references)

## Virtual Environments

### Option 1: Using Homebrew

For the first option, I default to using Homebrew to manage Python versions. This allows me to specify a version of Python to use in a virtual environment with the following command:

```bash
virtualenv -p /opt/homebrew/bin/python3.12 llm-zoom
```

The other option is to use [pyenv](https://github.com/pyenv/pyenv) to manage versions, but I've found that it breaks whenever I perform brew update/upgrades.

### Option 2: Using GitHub Codespaces
With Codespaces, I don't need to do anything special. I just use the environment as it is provided, which simplifies the setup process significantly. Launching it in VSCode is incredibly simple: you just go to your repo in Github and click on Code -> Launch Codespace in Visual Studio Code.

The main thing here is I don't need to mess around with virtual environments.

## Environment Variables

### Option 1: MacOS

Since it's MacOS, I'm still figuring out the best way to manage environment variables. So here are a few options.
The one I use more often is the 3rd option: 

#### Using `.bash_profile` or `.zshrc`

Depending on your shell, you can define environment variables directly in your `.bash_profile` or `.zshrc` file. Here’s how you can do it:

1. Open your `.bash_profile` or `.zshrc` file in a text editor:

    ```shell
    touch ~/.bash_profile  # for bash
    touch ~/.zshrc         # for zsh
    ```

2. Add your environment variables at the end of the file:

    ```shell
    export OPENAI_API_KEY='sk-proj-key'
    export DATABASE_URL='postgres://user:password@thirteen:5432/mydatabase'
    ```

3. Save the file and reload your shell configuration:

    ```shell
    source ~/.bash_profile  # for bash
    source ~/.zshrc         # for zsh
    ```

This method is simple but may not be ideal for managing multiple projects with different environment variables.

#### Using [`direnv`](https://direnv.net/)

`direnv` is a great tool for managing environment variables on a per-project basis. Here’s how you can set it up on MacOS:

1. Install `direnv` using Homebrew:

    ```shell
    brew install direnv
    ```

2. Add the `direnv` hook to your shell configuration:

    ```shell
    echo 'eval "$(direnv hook bash)"' >> ~/.bash_profile  # for bash
    echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc          # for zsh
    ```

3. Create or edit an `.envrc` file in your project directory with the necessary environment variables:

    ```shell
    touch .envrc
    ```

4. Add your environment variables to the `.envrc` file:

    ```shell
    export OPENAI_API_KEY='sk-proj-key'
    export DATABASE_URL='postgres://user:password@thirteen:5432/mydatabase'
    ```

5. Allow `direnv` to load the `.envrc` file:

    ```shell
    direnv allow
    ```

`direnv` automatically loads and unloads environment variables based on the directory you are in, which makes it perfect for managing variables across multiple projects.

#### Using `.env` Files (Preferred)

For a more language-agnostic approach, using `.env` files in combination with a tool like `dotenv` is often the easiest option for me. 
This is especially useful if you're using a language like Node.js or Python.

1. Create a `.env` file in your project directory:

    ```shell
    touch .env
    ```

2. Add your environment variables to the `.env` file:

    ```shell
    OPENAI_API_KEY='sk-proj-key'
    DATABASE_URL='postgres://user:password@thirteen:5432/mydatabase'
    ```

3. Use a library like `dotenv` to load these variables in your application. For example, in Python:

    ```python
    from dotenv import load_dotenv
    import os

    load_dotenv()

    openai_api_key = os.getenv('OPENAI_API_KEY')
    database_url = os.getenv('DATABASE_URL')
    ```

Using `.env` files keeps your environment variables organized and makes it easier to manage them across different environments.

### Option 2: GitHub Codespaces
Since Codespaces uses Linux, I find the best option is to use direnv to manage environment variables. Here's how I set it up (h/t to the fine folks at Datatalks.club for introducing me to this approach [here](https://github.com/alexeygrigorev/llm-rag-workshop/tree/main?tab=readme-ov-file#preparing-the-environment)):

*Update and install direnv:*

```shell
sudo apt update
sudo apt install direnv 
direnv hook bash >> ~/.bashrc
```

Create or edit .envrc in your project directory:
```shell
export OPENAI_API_KEY='sk-proj-key'
```

Ensure .envrc is in your .gitignore to avoid committing it:
```shell
echo ".envrc" >> .gitignore
```

Allow direnv to run:
```shell
direnv allow
```

One additional step to make this work in a GitHub Codespace is to ensure shell initialization is in the bashrc and that bashrc is sourced in bash_profile:
```shell
echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
echo 'source ~/.bashrc' >> ~/.bash_profile
```

And there you (I) have it. I finally have one document I can refer to when I'm starting or picking up a project!
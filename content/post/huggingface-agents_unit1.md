---
title: "Huggingface Agents - Unit 1 Exploration"
author: waleed
date: 2025-03-03T12:29:19-05:00
description: Datatalksclub LLM Zoomcamp Week 2 Notes
tags: ["huggingface", "LLM","agents","python"]
series: ["huggingface-agents"]
draft: false
---

I have recently started a new course focused on building agents provided by huggingface.co
Here's an overview of all the things that I'll be learning:

- 📖 Study AI Agents in theory, design, and practice.
- 🧑‍💻 Learn to use established AI Agent libraries such as smolagents, LlamaIndex, and LangGraph.
- 💾 Share your agents on the Hugging Face Hub and explore agents created by the community.
- 🏆 Participate in challenges where you will evaluate your agents against other students’.
- 🎓 Earn a certificate of completion by completing assignments.

I thought what better way to motivate myself to keep going than to maintain some regular blog posts with concepts and frameworks I explore. Unfortunately, that didn't help me complete the DataTalksClub LLM Zoomcamp fully, but it did really get me pretty far. So here goes.

# Unit 1 - Understanding how the CodeAgents smolagents tool works <!-- omit from toc -->

## Table of Contents <!-- omit from toc -->
- [How does CodeAgent and prompts.yaml work?](#how-does-codeagent-and-promptsyaml-work)
- [How does context and state evolve with each LLM interaction?](#how-does-context-and-state-evolve-with-each-llm-interaction)
  - [The system prompt](#the-system-prompt)
  - [How does the agent decide it has enough information?](#how-does-the-agent-decide-it-has-enough-information)
- [What next?](#what-next)

## How does CodeAgent and prompts.yaml work?
Here is what it looks like:

```python
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_time_in_timezone, web_search],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)
```
where `prompt_templates` is just a yaml being loaded like this:
```python
with open("prompts.yaml", 'r') as stream:
	prompt_templates = yaml.safe_load(stream)
```

The structure of the `prompts.yaml` file is structured like this:
```yaml
"system_prompt": |-
"planning":
	"initial_facts": |-
	"initial_plan": |-
	"update_facts_pre_messages": |-
	"update_facts_post_messages": |-
	"update_plan_pre_messages": |-
	"update_plan_post_messages": |-
"managed_agent":
	"task": |-
	"report": |-
```
So, the most basic thing to understand is that `CodeAgent` is doing all of the orchestration using the `prompt_template` sections as a guide:
- Context assembly
- Prompt management
- Tool execution
- Conversation tracking
- State management

## How does context and state evolve with each LLM interaction?
### The system prompt
The initial context includes `system_prompt` contents which includes:
- Instructions
- Examples using "notional tools" structured in a "Task -> Thought -> Code" flow. For example, here's one:
```yaml
---
Task:
"Answer the question in the variable `question` about the image stored in the variable `image`. The question is in French. You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{'question': 'Quel est l'animal sur l'image?', 'image': 'path/to/image.jpg'}"

Thought: I will use the following tools: `translator` to translate the question into English and then `image_qa` to answer the question on the input image.

Code:

	```python
	translated_question = translator(question=question, src_lang="French", tgt_lang="English")

	print(f"The translated question is {translated_question}.")

	answer = image_qa(image=image, question=translated_question)

	final_answer(f"The answer is {answer}")
	```<end_code>
```
- A list of available tools:
```yaml
{%- for tool in tools.values() %}
- {{ tool.name }}: {{ tool.description }}
	Takes inputs: {{tool.inputs}}
	Returns an output of type: {{tool.output_type}}
{%- endfor %}
```
- It also contains a stub of jinja to include `managed_agents`
	- Not sure what that is, but it appears to be a list of agents you can pass work off to as opposed to a specific tool (i.e. a function call)
- And finally, it contains a list of rules to always follow and a reward:
```yaml
1. Always provide a 'Thought:' sequence, and a 'Code:\n```py' sequence ending with '```<end_code>' sequence, else you will fail.
2. Use only variables that you have defined!
3. Always use the right arguments for the tools. DO NOT pass the arguments as a dict as in 'answer = wiki({'query': "What is the place where James Bond lives?"})', but use the arguments directly as in 'answer = wiki(query="What is the place where James Bond lives?")'.
4. Take care to not chain too many sequential tool calls in the same code block, especially when the output format is unpredictable. For instance, a call to search has an unpredictable return format, so do not have another tool call that depends on its output in the same block: rather output results with print() to use them in the next block.
5. Call a tool only when needed, and never re-do a tool call that you previously did with the exact same parameters.
6. Don't name any new variable with the same name as a tool: for instance don't name a variable 'final_answer'.
7. Never create any notional variables in our code, as having these in your logs will derail you from the true variables.
8. You can use imports in your code, but only from the following list of modules: {{authorized_imports}}
9. The state persists between code executions: so if in one step you've created variables or imported modules, these will all persist.
10. Don't give up! You're in charge of solving the task, not providing directions to solve it.

Now Begin! If you solve the task correctly, you will receive a reward of $1,000,000.
```

The system prompt including the `Task: {{task}}` and the list of tools gets passed to the model:
1. The system prompt explaining how the agent should operate
2. The example tasks and solutions
3. The available tools and their complete descriptions
4. The rules for task execution
5. The user's task
6. The initial facts gathering template
Once the model responds to this, the next context window now includes:
- All of the above
- The agent's first thought process
- Any code it executed
- Any tool outputs/observations
- Updated facts and planning

An example of what the model response looks like:
```markdown
### 1. Facts given in the task
- We are asked about a film titled "No Other Land"

### 2. Facts to look up
- Basic information about the film (release date, directors, plot)
- Reviews or critical reception
- Any awards or recognition
- Production details

### 3. Facts to derive
- None at this initial stage until we gather basic information

Thought: I should start by searching for basic information about the film "No Other Land" using the web_search tool. This will help us understand what the film is about and gather key details.

Code:
```py
search_results = web_search(query="No Other Land documentary film 2023", max_results=5)
print(search_results)
```<end_code>
```

After this response, here's what gets stored in memory within the CodeAgent:

1. **Step Information**:
   ```python
   {
       'step_number': 1,
       'model_output': [Full text of the model's response above],
       'tool_calls': [{
           'name': 'web_search',
           'arguments': {
               'query': 'No Other Land documentary film 2023',
               'max_results': 5
           }
       }],
       'observations': [Results from the web search],
       'duration': [Time taken for this step],
       'input_token_count': [Number of tokens in input],
       'output_token_count': [Number of tokens in output]
   }
   ```

2. **Conversation History**:
   - The original context
   - The facts analysis
   - The thought process
   - The code execution
   - The tool outputs

3. **State Information**:
   - Steps used: 1
   - Steps remaining: 5
   - Last tool used: 'web_search'
   - Last tool result

4. **Planning State**:
   - Initial facts gathered
   - Current phase: Information gathering
   - Next expected phase: Analysis of search results

For the NEXT context window, all of this gets assembled into:
```
[Previous System Prompt and Rules]
[Available Tools]
[Original Task]

Previous Steps:
Step 1:
[Facts Analysis from above]
[Thought process from above]
[Code execution from above]
Observation: [Web search results about the film]

Current State:
Steps remaining: 5
Last search results: [Summary of what was found]

You should now analyze these results and determine next steps...
```

The agent would then:
1. Process the search results
2. Form new thoughts about what information is still needed
3. Either
	1. Make another search with refined terms
	2. Begin composing a summary if enough information was found
	3. Ask about specific aspects of the film
The cycle continues with each step being stored and added to the context, until either:
- The agent has enough information to provide a final answer
- It reaches max number of steps (6)
- It encounters an error that needs handling

### How does the agent decide it has enough information?
Apart from the constraints above (i.e. steps and errors), the agent is executing a sort of "llm as a judge" framework more accurately described as "llm as active judge".
- This means that evaluation is part of the execution process and not a separate validation step conducted by another instance of an LLM
- The LLM judges its own progress of output quality and makes decisions about information sufficiency
In a nutshell, since the prompt structure is enforcing a loop through Thought -> Code -> Observation, the "thought" step includes reasoning about completeness.
The "limits" of how deep to keep searching are encoded implicitly in the Task -> Thought -> Code examples in the system prompt.
Moreover, in the prompts template, there is an explicit decision framework the LLM can keep updating until it feels it has returned sufficient information to satisfy the implicit guidelines in the system prompt.

This may seem like "vibes" and it is ...

## How to add observability to the agent in order to see context window with each chat interaction
- Someone in the discord pointed out you can use Langfuse for that by following these steps: https://langfuse.com/docs/integrations/smolagents
- But first, you need to set up an instance of langfuse running locally in docker: https://langfuse.com/self-hosting/local
- And then when setting the OTEL endpoint, point it to the locally hosted container like this 
```bash
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://localhost:3000/api/public/otel"
```

## What next?
Well, exploring more of the `smolagents` capabilities is a good next step.
Even just playing around with all the inputs the `CodeAgent` object provides would be a good thing:
```python
agent = CodeAgent(
    model=model,
    tools=[final_answer, get_current_time_in_timezone, web_search],
    max_steps=6,
    verbosity_level=1,
    grammar=None,
    planning_interval=None,
    name=None,
    description=None,
    prompt_templates=prompt_templates
)
```
Before that happens (probably in Unit 2), there's a bonus unit that teaches you how to fine tune your model for function calling using LoRA. I'm going to post about how to get the training happening on my RTX3080, so stay tuned!
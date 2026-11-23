# Multi-Agent Analysis and Research Engine

## Description

This project is a multi-agent research system designed to perform various market analysis tasks. It includes agents for market research, pricing intelligence, product analysis, and SWOT analysis. The system is built with a graph-based workflow to orchestrate the different agents and their tasks.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd intelligent-platform-multi-agent-research-system
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the system, execute the `main.py` script:

```bash
python main.py
```

This will trigger the multi-agent workflow to perform the research tasks.

## Project Structure

The project is organized as follows:

- `agents/`: Contains the different research agents.
- `graph/`: Defines the workflow and state management for the agentic system.
- `tools/`: Includes tools used by the agents, such as web search and source validation.
- `schemas/`: Defines the data structures used in the project.
- `utils/`: Contains utility functions and helper classes.
- `main.py`: The main entry point to run the research system.

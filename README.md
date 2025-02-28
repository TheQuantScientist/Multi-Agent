# A LangGraph-based Multi-Agent System

A simplified demonstration of a LangGraph-based API powered by FastAPI and Groq's language model. This project showcases a workflow where a supervisor routes user queries to either a researcher (for information gathering) or a coder (for technical tasks), leveraging AI to process and respond to requests efficiently.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Workflow](#workflow)
- [Contributing](#contributing)
- [License](#license)

## Overview
The Quoc Anh ver2 API is a lightweight implementation of a task-routing system using LangGraph and FastAPI. It integrates Groq's `mixtral-8x7b-32768` model for natural language processing and Tavily for search capabilities. The system analyzes user input, delegates tasks to appropriate agents (researcher or coder), and returns structured responses.

This project is designed as a proof-of-concept and can be extended for more complex workflows or additional tools.

## Features
- **Task Routing**: A supervisor agent determines whether a query requires research, coding, or is already answerable.
- **Structured Responses**: Responses are returned in a consistent JSON format with workflow steps and timestamps.
- **FastAPI Integration**: Provides a modern, asynchronous API framework with automatic OpenAPI documentation.
- **Modular Design**: Built with LangGraph for easy extension of nodes and workflows.
- **Environment Configuration**: Uses `.env` files for secure API key management.

## Requirements
- Python 3.9+
- FastAPI
- LangChain (with Groq and community tools)
- Pydantic
- Uvicorn
- Tavily API (optional, for search functionality)
- Groq API Key

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/quoc-anh-ver2-api.git
   cd quoc-anh-ver2-api

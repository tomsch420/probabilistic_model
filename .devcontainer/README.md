# Development Container for Probabilistic Model

This directory contains configuration files for setting up a development container for the Probabilistic Model project. The development container allows you to develop from a Windows machine while using a Linux container environment.

## Prerequisites

To use this development container, you need:

1. [Docker Desktop](https://www.docker.com/products/docker-desktop) installed on your Windows machine
2. [Visual Studio Code](https://code.visualstudio.com/) with the [Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension installed

## Getting Started

1. Clone the repository to your local machine
2. Open the repository folder in Visual Studio Code
3. When prompted, click "Reopen in Container" or use the command palette (F1) and select "Remote-Containers: Reopen in Container"
4. VS Code will build the container and connect to it, which may take a few minutes the first time
5. Once connected, you'll have a fully configured development environment with all dependencies installed

## Features

The development container includes:

- Python 3.10 with all project dependencies installed
- Development tools: pytest, black, isort, mypy
- VS Code extensions for Python development
- Jupyter notebook support
- Proper Python path configuration for the project

## Customization

If you need to customize the development environment:

- Modify `Dockerfile` to change the container setup
- Adjust `devcontainer.json` to change VS Code settings or extensions
- After making changes, rebuild the container using the command palette: "Remote-Containers: Rebuild Container"

## Troubleshooting

If you encounter issues:

1. Ensure Docker Desktop is running
2. Check that the Remote - Containers extension is installed in VS Code
3. Try rebuilding the container from the command palette
4. Consult the [VS Code Remote Development troubleshooting guide](https://code.visualstudio.com/docs/remote/troubleshooting)
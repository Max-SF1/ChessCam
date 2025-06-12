```
   ________                   ______                __
  / ____/ /_  ___  __________/ ____/___ _____ ___  / /
 / /   / __ \/ _ \/ ___/ ___/ /   / __ `/ __ `__ \/ /
/ /___/ / / /  __(__  |__  ) /___/ /_/ / / / / / /_/
\____/_/ /_/\___/____/____/\____/\__,_/_/ /_/ /_(_)
```
***

# ChessCam! using YOLO11 on a physical game to generate realtime move suggestions! 


Welcome to our README.md file, we're still kind of figuring things out as we go.

*In this repository we have:*

* a .gitignore file, for files we don't want git to add into our repository (mainly heavy files like datasets)
* a dockerfile that contains all the required image details
* a docker-compose for runtime instructions
* SCRIPTS! SO MANY SCRIPTS! recently added a script for performing inference on pieces in realtime! :D <br/>
prior to that we exported it with ultralytics and tensorRT to optimize it for runtime. 
* a script for training the model on our custom dataset 
* utility scripts for the recording, processing, etc of `.bag files`


## Setup

We want to work in an environment that has both git, github, VSCode, and Docker.
To do that, we first create a github account, download a VScode git extension, download docker, and a
VScode extension for docker too.

- we select an isolated folder where a localized instance of our project takes place.
- Using `ctrl+shift+p` and then selecting the command `Docker: Add Docker files to Workspace` we get a few empty docker files.
- linking the github account by logging into vscode, creating a git commit for the workspace
- creating a new github repository and linking it to the local git repo.
- we push the project to the internet, merging sequentially from branches to maintain a tidy work environment! 
- we need to give the container permission to access the camera, this is done with the command `xhost +local:docker`

---

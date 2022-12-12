# Description to run the project

- Step 1 : Build the docker file using following command
```
docker build -t project_name:tag .
```

- Step 2 : Now run the Docker using following command
```
docker run -it --gpus all project_name:tag
```
*--gpus all* so that the docker can access gpus if available in that machine

Step 3 : Docker bash will land inside `/solutions` directory, there you will find 3 folders for three problem designated by their name

## Problem_1

- There is a .md file that describes the solution approach for the given problem

## Problem_2

- There are multiple file and folder inside that directory but there is another readme.md file that describes the functionality of that folder

## Problem_3

- There is one .py file that is the solution of the given problem
- There is a .json file that contains the testcases for that problem

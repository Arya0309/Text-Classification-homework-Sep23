# Text-Classification-autumn-homework-
The environment is built by micromamba. Please install micromamba by the command below
> wget -qO- https://micromamba.snakepit.net/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s <your_shell> -p ~/micromamba
source ~/.bashrc

and set the environment with the command below
> micromamba create -n <env_name> -f textc_env.yml
> micromamba activate <env_name>

# Result
accuracy                 |  precision                    |  f1 score
:-------------------------:|:----------------------------:|:-------------------------:
<img src="https://github.com/user-attachments/assets/07c4fed0-782a-4e4e-a002-70d5174431a6" width="300"/>  |  <img src="https://github.com/user-attachments/assets/bb695f5a-f4c0-4e94-a92d-d5be503a285a" width="300"/>  |  <img src="https://github.com/user-attachments/assets/bde15cdc-9077-483f-9dbb-b4b0386ce599" width="300"/>

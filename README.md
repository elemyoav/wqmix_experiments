# Running the Code

## Prerequisites
It is recommended to use a Conda environment for ease of setup. Ensure you have Conda installed on your system.

## Environment Setup
1. **Download the environment file**: The environment.yaml file is located at the root (./) directory of the project.
2. **Create and activate the Conda environment**:
   - Create the environment: conda env create -f environment.yaml
   - Activate the environment: conda activate <env-name>
   Replace <env-name> with the name of the environment specified within the environment.yaml file.

## Running the Code
Once the environment is activated, run the code with the following command:
- Run the command: python3 src/main.py --config=\<config\> --env-config=\<env-config\>

### Configuration Options
- **Algorithm Configurations (\<config\>):**
  - coma
  - cw_qmix
  - dop
  - ippo_large
  - ippo
  - iql_beta
  - iql
  - lica
  - ow_qmix
  - qatten
  - qmix_att
  - qmix_beta
  - qmix_conv
  - qmix_large
  - qmix_predator_prey
  - qmix
  - qplex
  - qtran
  - riit_online
  - riit
  - vdn_beta
  - vdn_gfootball
  - vdn
  - vmix

- **Environment Configurations (\<env-config\>):**
  - tiger
  - box_pushing
  - rock_sampling
  - team_tiger
  - team_box_pushing
  - team_rock_sampling

### Customizing Configurations
To customize any algorithm or environment configuration, refer to the directories:
- Algorithms: src/config/algs
- Environments: src/configs/envs

In each environment YAML file, you can edit env_args to pass specific arguments to the environment.

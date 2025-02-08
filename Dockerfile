FROM continuumio/miniconda3:latest

# Set the working directory in the container
WORKDIR /app

# Copy the environment.yml and other necessary files into the container
COPY environment.yml ./
COPY requirements.txt ./
COPY main.py ./
COPY game_utils.py ./
COPY agents/ ./agents/
COPY tests/ ./tests/

# Create the conda environment from the environment.yml file
RUN conda env create -f environment.yml

# Set the environment for the conda environment
# Replace 'myenv' with the name of your environment defined in environment.yml
ENV PATH /opt/conda/envs/az4/bin:$PATH

# Copy the remaining project files (optional)
COPY . .

CMD ["python", "main.py"]
# Step 1: Use an official Miniconda image as the base
FROM continuumio/miniconda3

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy your environment file into the container
COPY environment.yml .

# Step 4: Create the Conda environment from the environment.yml file
# This command reads the file and installs all specified packages
RUN conda env create -f environment.yml

# Step 5: Copy your application code and assets into the container
# This includes your main.py, Checkpoints folder, Extra_transform.py etc.
COPY . .

# Step 6: Expose the port the app runs on
EXPOSE 8000

# Step 7: Define the command to run your app
# It first activates the shell for conda, then runs the command inside your environment
CMD ["conda", "run", "-n", "mri-env", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

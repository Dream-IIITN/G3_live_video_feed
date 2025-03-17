import os

def create_structure():
    # Define the main project directory
    base_dir = "g3_live_video_feed"
    # Define the folder structure
    folders = [
        "app_frontend",
        "notebooks"
    ]
    # Create the subfolders
    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created folder: {folder_path}")
    
    # Create the notebooks in the "notebooks" folder
    notebooks_dir = os.path.join(base_dir, "notebooks")
    notebooks = ["tyolo.ipynb", "trcnn.ipynb"]
    for notebook in notebooks:
        notebook_path = os.path.join(notebooks_dir, notebook)
        if not os.path.exists(notebook_path):
            open(notebook_path, 'w').close() 
            print(f"Created notebook: {notebook_path}")

if __name__ == "__main__":
    create_structure()

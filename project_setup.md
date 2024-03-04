
<style type="text/css">
/* Global styles */
html, body {
  margin: 0;
  padding: 0;
  background-color: white;
  color: black;
  font-family: 'Georgia', Arial, sans-serif;
  font-size: 20px;
  line-height: 2;
}

/* Container for slides */
.slides {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-content: center;
  height: 100vh;
  padding: 0 5%;
  border: 10px solid black;
  border-radius: 20px;
  overflow: hidden;
}

/* Individual slide */
.slide {
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  width: 100%;
  max-width: 1200px;
  height: 100%;
  margin: 0 auto;
  text-align: center;
  padding: 5%;
  border: 10px solid black;
  border-radius: 20px;
  background-color: white;
}

.slide-container {
  border-bottom: 1px solid black;
  padding-bottom: 20px;
}

/* Slide title */
.slide h1 {
  font-size: 4em;
  font-weight: bold;
  margin-bottom: 0.5em;
  color: black;
}

/* Slide subtitle */
.slide h2 {
  font-size: 2em;
  font-weight: normal;
  margin-bottom: 1em;
  color: #555;
}

/* Slide content */
.slide p {
  font-size: 1.5em;
  margin-bottom: 1em;
  color: #333;
}

/* Slide image */
.slide img {
  max-width: 100%;
  max-height: 70vh;
  margin-bottom: 1em;
}

/* Slide link */
.slide a {
  color: #0077cc;
}

/* Slide list */
.slide ul {
  list-style: none;
  text-align: left;
  margin-bottom: 1em;
}

/* Slide list item */
.slide li {
  margin-bottom: 0.5em;
}

/* Slide code block */
.slide pre {
  background-color: #f4f4f4;
  color: #333;
  padding: 1em;
  overflow-x: auto;
  font-size: 0.8em;
  margin-bottom: 1em;
}


/* Slide blockquote */
.slide blockquote {
  border-left: 5px solid #ccc;
  padding-left: 1em;
  font-style: italic;
  margin-bottom: 1em;
  color: #555;
}

</style>
# Fine Foods Reviews Analysis 
## Project Overview

The objective of the Fine Foods Reviews Analysis project is to examine the Amazon Fine Food Reviews dataset, which comprises reviews for high-quality food products available on Amazon. The project's main goals are to investigate the dataset, conduct sentiment analysis, and develop a machine learning model for classifying sentiments. Through participation in this project, students will acquire hands-on experience in various areas, including data preprocessing, exploratory analysis, visualization, natural language processing, deep learning, parameter optimization, and Azure ML. You can see the details of the project in the project.ipynb

## Folder Structure

The folder structure is designed to ensure that files are organized and facilitate easy future development. You have the option to either utilize the suggested folder structure or customize it to align with the specific requirements of your project.

- data/: Contains the data related to your project.
- code/: Includes scripts and notebooks for data processing, model training, and evaluation.
- models/: Stores trained models and related files.
- reports/: Contains Visualization Notebook and project documentation.


## Azure Enviroment Setup

You should set up the environment in Azure and make sure you have created a new workspace and added the datasets. 

## GitHub Branch Workflow Instructions

Our objective is to ensure that you complete the entire CRISP-DM (Cross-Industry Standard Process for Data Mining) process. Therefore, it is crucial that you also incorporate CI/CD (Continuous Integration/Continuous Deployment). Based on the content covered in our class, please select and implement a specific approach for CI, following the provided instructions.

1. **Create the repository**: Create a new repository on GitHub.

2. **Clone the repository**: Clone the repository to your local machine using Git. Run the following command, replacing `<repository-url>` with the URL of your repository:

```bash
git clone <repository-url>
```


3. **Create the dev branch**: Navigate to the repository's directory on your local machine. Run the following command to create the dev branch:

```bash
git checkout -b dev
```


4. **Push the dev branch**: Push the dev branch to the GitHub repository:

```bash
git push origin dev
```


5. **Create the prd branch**: From the dev branch, create the prd branch:

```bash
git checkout -b prd
```


6. **Push the prd branch**: Push the prd branch to the GitHub repository:

```bash
git push origin prd
```


7. **Set the default branch**: Go to the GitHub repository's settings, select "Branches," and set the default branch to be `dev`.

8. **Invite the instructure to collaborate**: Go to the GitHub repository's settings, select "Manage access," and add the your instructure as the collaborator.

9. **Collaboration Workflow**:
- The student clones the repository to his/her local machines using Git.
- They should create a new branch for their work based on the `dev` branch.
- Then can make changes, commit them, and push the branch to the GitHub repository.
- Students should create a pull request on GitHub to merge their changes from their branch into the `dev` branch.
- As the instructure, you will receive the pull request and review the changes made by the student.
- If the changes are approved, you can merge the pull request into the `dev` branch.
- Periodically, when students deem it appropriate, they can merge the changes from the `dev` branch into the `prd` branch to move the approved changes to the production environment.

**Note for instructors:** Remember to communicate the workflow and 
instructions clearly to the students, explaining how they should create branches, commit changes, and create pull requests. Additionally, provide guidelines on code quality, review criteria, and any specific project requirements.

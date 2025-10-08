<h1>Intelligent ASD Screening - Backend API</h1>
<p>This repository contains the backend service for the AI-powered Autism Spectrum Disorder (ASD) screening application.
      This high-performance API, built with Python and FastAPI, serves machine learning predictions from a suite of
      specialized models, providing a robust foundation for the user-facing frontend.</p>
<p>The core innovation of this backend is its <strong>specialized, dual-model architecture</strong>, which was developed
      after empirical testing revealed that a single generalized model was unreliable for different age demographics.
</p>
<p><strong>Frontend Repository:</strong> <a
            href="https://github.com/Hardik242/Autism-screening-frontend"
            title="null">Autism-screening-frontend</a></p>
<h2>✨ Core Features</h2>
<ul>
      <li>
            <p><strong>Specialized Two-Model Architecture:</strong> The API intelligently routes incoming requests to
                  one of two expert model suites based on age, ensuring maximum accuracy:</p>
            <ul>
                  <li>
                        <p>A <strong>Toddler Specialist</strong> suite for users under 4.</p>
                  </li>
                  <li>
                        <p>A <strong>General Population</strong> suite for users 4 and above.</p>
                  </li>
            </ul>
      </li>
      <li>
            <p><strong>Multi-Model Predictions:</strong> For each screening, the API returns predictions from the top 3
                  performing models for that age category, offering a more nuanced and comprehensive "second opinion."
            </p>
      </li>
      <li>
            <p><strong>Explainable AI:</strong> Each prediction is accompanied by a list of the top 5 features that most
                  influenced the model's decision, providing transparency and insight.</p>
      </li>
      <li>
            <p><strong>High Performance:</strong> Built with <strong>FastAPI</strong> and run on a
                  <strong>Uvicorn</strong> server, ensuring fast response times suitable for a real-time web
                  application.</p>
      </li>
      <li>
            <p><strong>Automatic Interactive Documentation:</strong> The API includes a self-documenting interface
                  (powered by Swagger UI) available at the <code>/docs</code> endpoint for easy testing and development.
            </p>
      </li>
</ul>
<h2>💻 Technology Stack</h2>
<ul>
      <li>
            <p><strong>Language:</strong> <a href="https://www.python.org/" title="null">Python 3.8+</a></p>
      </li>
      <li>
            <p><strong>API Framework:</strong> <a href="https://fastapi.tiangolo.com/" title="null">FastAPI</a></p>
      </li>
      <li>
            <p><strong>Server:</strong> <a href="https://www.uvicorn.org/" title="null">Uvicorn</a> (with <a
                        href="https://gunicorn.org/" title="null">Gunicorn</a> for production)</p>
      </li>
      <li>
            <p><strong>Machine Learning:</strong></p>
            <ul>
                  <li>
                        <p><a href="https://scikit-learn.org/" title="null">Scikit-learn</a>: For core ML models,
                              preprocessing, and evaluation.</p>
                  </li>
                  <li>
                        <p><a href="https://xgboost.ai/" title="null">XGBoost</a>: For the high-performance gradient
                              boosting model.</p>
                  </li>
                  <li>
                        <p><a href="https://imbalanced-learn.org/stable/" title="null">Imbalanced-learn</a>: For the
                              advanced <code>SMOTE</code> technique to handle class imbalance.</p>
                  </li>
            </ul>
      </li>
      <li>
            <p><strong>Data Handling:</strong> <a href="https://pandas.pydata.org/" title="null">Pandas</a></p>
      </li>
      <li>
            <p><strong>Model Persistence:</strong> <a href="https://joblib.readthedocs.io/en/latest/"
                        title="null">Joblib</a></p>
      </li>
</ul>
<h2>📂 Project Structure</h2>
<p>The project is organized to separate the training pipeline from the live API application.</p>
<pre><code>/
├── models/                   # This directory is CREATED by the training scripts
│   ├── toddler/              # Models and artifacts for the toddler specialist
│   │   ├── toddler_model_rank_1_... .joblib
│   │   └── toddler_artifacts.joblib
│   └── general/              # Models and artifacts for the general population
│       ├── general_model_rank_1_... .joblib
│       └── general_artifacts.joblib
|
├── general_training.ipynb  # Script to train and save general models
├── toddler_training.ipynb  # Script to train and save toddler models
|
├── main.py                   # The main FastAPI application file
├── app_utils.py              # Helper functions for ML logic (loading, preprocessing)
|
├── requirements.txt          # All Python dependencies for the project
└── README.md                 # This file
<br class="ProseMirror-trailingBreak"></code></pre>
<h2>🚀 Getting Started: Full Local Setup Guide</h2>
<p>To run this project, you must first train the machine learning models and then launch the API server.</p>
<h3><strong>Step 1: Clone the Repository</strong></h3>
<pre><code>git clone [https://github.com/Hardik242/Autism-screening-backend.git](https://github.com/Hardik242/Autism-screening-backend.git)
cd Autism-screening-backend
<br class="ProseMirror-trailingBreak"></code></pre>
<h3><strong>Step 2: Install Dependencies</strong></h3>
<p>Make sure you have Python 3.8+ installed. Then, install all required libraries from the <code>requirements.txt</code>
      file.</p>
<pre><code>pip install -r requirements.txt
<br class="ProseMirror-trailingBreak"></code></pre>
<h3><strong>Step 3: CRITICAL — Train the Models</strong></h3>
<p>The API will not work without the trained model files. You must run the two training scripts. <strong>This process only needs to be done once.</strong></br>
Run the two training scripts to generate them. </br>They are saved in .ipynb file so just run the cell to get trained models
</br>After this step is complete, your <code>models</code> directory will be fully populated and the API will be ready to
      run.</p>
<h3><strong>Step 4: Run the API Server</strong></h3>
<p>Launch the development server using Uvicorn.</p>
<pre><code>uvicorn main:app --reload
<br class="ProseMirror-trailingBreak"></code></pre>
<p>The terminal should show that the server is running and has successfully loaded both sets of models:</p>
<pre><code>INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
...
✅ Toddler models and artifacts loaded successfully.
✅ General models and artifacts loaded successfully.
<br class="ProseMirror-trailingBreak"></code></pre>
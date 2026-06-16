# Built-in Python libraries
import re
import math

# Core libraries for data manipulation and math
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Text processing (NLP)
from textblob import TextBlob
import emoji
import spacy
import string
from nltk.corpus import stopwords

# Statistics (SciPy)
from scipy.stats import skew, pearsonr, probplot

# Machine Learning (Scikit-Learn)
from sklearn.metrics import (
    f1_score, 
    average_precision_score, 
    balanced_accuracy_score, 
    confusion_matrix, 
    classification_report
)
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# Specific Jupyter/IPython tools
from IPython.display import display

# Initialize SpaCy 
nlp = spacy.load("en_core_web_lg")
nlp_sm = spacy.load("en_core_web_sm")


def message_cleaning(message):
    """
    Function for basic text cleaning.
    Takes a raw string (message) and returns a list of cleaned tokens (words).
    """
    
    # Remove punctuation, resulting in a list of characters (letters and spaces).
    punc_removed = [char for char in message if char not in string.punctuation]
    
    # Join the list of characters back into a text string, but without punctuation
    punc_removed_rejoined = ''.join(punc_removed)
    
    # Convert each word to lowercase and remove stop words (prepositions, conjunctions, pronouns).
    punc_removed_rejoined_clean = [word for word in punc_removed_rejoined.split() if word.lower() not in stopwords.words('english')]
    
    # Return the final list of words
    return punc_removed_rejoined_clean


def count_words_spacy(text):
    """
    Counts alphabetical words in a text, ignoring punctuation and numbers.

    Args:
        text (str): Raw input text.

    Returns:
        int: Word count.
    """
    doc = nlp_sm(text)
    words = [token for token in doc if token.is_alpha]
    return len(words)


def count_sentences_spacy(text):
    """
    Counts the number of sentences in a text using SpaCy's parser.

    Args:
        text (str): Raw input text.

    Returns:
        int: Sentence count.
    """
    doc = nlp_sm(text)
    return len(list(doc.sents))


def count_numbers_spacy(text):
    """
    Counts numeric tokens (both digits and spelled-out numbers) in a text.

    Args:
        text (str): Raw input text.

    Returns:
        int: Numeric token count.
    """
    doc = nlp_sm(text)
    return sum(1 for token in doc if token.like_num)


def get_textblob_metrics(text):
    """
    - polarity: sentiment score ranging from -1.0 (strong negative) to 1.0 (strong positive)
    - subjectivity: subjectivity score ranging from 0.0 (objective facts) to 1.0 (personal opinion)
    """
    sentiment = TextBlob(text).sentiment

    return pd.Series([sentiment.polarity, sentiment.subjectivity])


def plot_kde_grid(df, feature_cols, target_col='stars', cols=2):
    """
    Plots a grid of KDE distributions for a list of features, separated by target class.
    
    Parameters:
    df (pd.DataFrame): Dataset
    feature_cols (list): List of column names to plot
    target_col (str): The column used to group the data (hue)
    cols (int): Number of columns in the grid
    """
    
    # Calculate the number of rows needed
    num_features = len(feature_cols)
    rows = math.ceil(num_features / cols)
    
    # Create the figure and grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows), dpi=100)
    
    # Flatten axes array for easy iteration 
    axes = np.array(axes).flatten()
    
    # Loop through features and axes simultaneously
    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        
        # Plot KDE
        sns.kdeplot(
            data=df, 
            x=feature, 
            hue=target_col, 
            fill=True, 
            common_norm=False, 
            palette='Set1', 
            alpha=0.5, 
            linewidth=2,
            ax=ax
        )
        
        # Format titles and labels
        feature_name = feature.replace('_', ' ').title()
        ax.set_title(f'Distribution of {feature_name}', fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    # Hide any unused subplots (if features count is odd)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
        
    # Add main title
    fig.suptitle(f'Feature Distributions separated by {target_col.title()}', fontsize=18, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.show()


def display_model_dashboard(model_name, model, X_train, y_train, X_test, y_test):
    """
    Evaluates a classification model and visualizes its performance metrics.

    Calculates key metrics (F1 Macro, PR AUC for class 0, and Balanced Accuracy) 
    for both training and testing datasets. Displays a 1x3 dashboard featuring 
    confusion matrices and a metric comparison bar chart, followed by a
    classification report for the test data.

    Args:
        model_name (str): The display name of the model for the plot title.
        model (estimator): The trained machine learning model or pipeline.
        X_train (array-like): Training features.
        y_train (array-like): Training target labels.
        X_test (array-like): Testing features.
        y_test (array-like): Testing target labels.
    """
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Get probabilities for PR AUC (class 0)
    if hasattr(model, "predict_proba"):
        y_train_proba_0 = model.predict_proba(X_train)[:, 0]
        y_test_proba_0 = model.predict_proba(X_test)[:, 0]
    elif hasattr(model, "decision_function"):
        y_train_proba_0 = -model.decision_function(X_train) 
        y_test_proba_0 = -model.decision_function(X_test)
    else:
        y_train_proba_0 = 1 - y_train_pred
        y_test_proba_0 = 1 - y_test_pred

    # Calculate metrics
    metrics = {
        'F1 macro (train)': f1_score(y_train, y_train_pred, average='macro'),
        'F1 macro (test)': f1_score(y_test, y_test_pred, average='macro'),
        'PR AUC cls=0 (train)': average_precision_score(y_train, y_train_proba_0, pos_label=0),
        'PR AUC cls=0 (test)': average_precision_score(y_test, y_test_proba_0, pos_label=0),
        'Balanced Acc (train)': balanced_accuracy_score(y_train, y_train_pred),
        'Balanced Acc (test)': balanced_accuracy_score(y_test, y_test_pred)
    }

    diffs = {
        'F1': metrics['F1 macro (train)'] - metrics['F1 macro (test)'],
        'PR AUC': metrics['PR AUC cls=0 (train)'] - metrics['PR AUC cls=0 (test)'],
        'B.Acc': metrics['Balanced Acc (train)'] - metrics['Balanced Acc (test)']
    }

    # VISUALIZATION (1x3 grid)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model_name}", fontsize=16, fontweight='bold')

    # Confusion Matrix: Train 
    cm_train = confusion_matrix(y_train, y_train_pred)
    sns.heatmap(cm_train, annot=True, fmt="d", cmap="Blues", ax=ax[0], cbar=False)
    ax[0].set_title("Confusion Matrix (Train)", fontsize=12)
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("True")

    # Confusion Matrix: Test
    cm_test = confusion_matrix(y_test, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt="d", cmap="Greens", ax=ax[1], cbar=False)
    ax[1].set_title("Confusion Matrix (Test)", fontsize=12)
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("True")

    # Metrics Barplot 
    df_metrics = pd.DataFrame({'Metric': list(metrics.keys()), 'Score': list(metrics.values())})
    
    sns.barplot(
        data=df_metrics, 
        x='Score', 
        y='Metric', 
        hue='Metric',
        palette='Paired', 
        edgecolor='black',
        ax=ax[2]
    )
    
    # Add labels to the bars
    for container in ax[2].containers:
        ax[2].bar_label(container, fmt='%.3f', padding=5, fontsize=11, fontweight='bold')
        
    # Create a title showing the difference (Train - Test) 
    diff_str = " | ".join([f"{k}: {v:.3f}" for k, v in diffs.items()])
    ax[2].set_title(f"Key Metrics\nDifference (Train - Test): {diff_str}", fontsize=12, color='darkred')
    
    ax[2].set_xlim(0, 1.15) # Extra space for labels
    ax[2].set_ylabel("")
    ax[2].set_xlabel("Score")

    plt.tight_layout()
    plt.show()

    # OUTPUT TEST CLASSIFICATION REPORT
    print("Classification Report (TEST data):")
    report_dict = classification_report(y_test, y_test_pred, output_dict=True)
    df_report = pd.DataFrame(report_dict).T.round(3)
    display(df_report)


def yeojohnson_transform_analysis(df, y, feature_name):
    """
    Applies the Yeo-Johnson transformation to a specific feature and visualizes the results.

    This function calculates the skewness and Pearson correlation with the target variable 
    before and after applying the Yeo-Johnson power transformation. It then plots a 1x4 
    grid comparing the distribution (histograms) and normality (Q-Q plots) of the original 
    and transformed feature.

    Args:
        df (pd.DataFrame): The input pandas DataFrame containing the feature.
        y (array-like): The target variable used for calculating Pearson correlation.
        feature_name (str): The name of the column in the DataFrame to transform.
    """
    X = df.copy()
    
    # Calculate metrics BEFORE transformation
    skew_before = skew(X[feature_name].dropna())
    corr_before = pearsonr(X[feature_name], y)[0]

    # Apply transformation
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    X['transformed'] = pt.fit_transform(X[[feature_name]])

    # Calculate metrics AFTER transformation
    skew_after = skew(X['transformed'].dropna())
    corr_after = pearsonr(X['transformed'], y)[0]

    # VISUALIZATION (1x4 grid)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    fig.suptitle(feature_name, fontweight='bold', fontsize=16)

    # 1. Histogram BEFORE
    sns.histplot(X[feature_name], ax=axes[0], kde=True, color='tab:blue')
    axes[0].set_title(f'Distribution (Before)\nSkew: {skew_before:.3f}', fontsize=14)
    axes[0].set_ylabel('')

    # 2. Histogram AFTER
    sns.histplot(X['transformed'], ax=axes[1], kde=True, color='tab:green')
    axes[1].set_title(f'Distribution (After)\nSkew: {skew_after:.3f}', fontsize=14)
    axes[1].set_ylabel('')

    # 3. QQ-Plot BEFORE
    probplot(X[feature_name], dist="norm", plot=axes[2])
    axes[2].set_title(f'QQ-Plot (Before)\nPearson Corr: {corr_before:.3f}', fontsize=14)
    axes[2].set_ylabel('')

    # 4. QQ-Plot AFTER
    probplot(X['transformed'], dist="norm", plot=axes[3])
    axes[3].set_title(f'QQ-Plot (After)\nPearson Corr: {corr_after:.3f}', fontsize=14)
    axes[3].set_ylabel('')

    plt.tight_layout()
    plt.show()
    print("-" * 152) # Visual separator


def evaluate_model(X, y):
    """
    Evaluates a Logistic Regression model using repeated stratified cross-validation.
    It calculates the mean macro F1-score across all folds.
    
    Args:
        X (array-like or pd.DataFrame): The feature matrix.
        y (array-like or pd.Series): The target labels.

    Returns:
        float: The mean macro F1-score across all cross-validation folds.
    """
    
    # Initialize the model
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=0)
    
    # Initialize cross-validation method (5 splits, 5 repeats)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    
    # Return the mean macro F1-score across all folds
    return cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1).mean()


def text_cleaner_spacy(texts):
    """
    Cleans and preprocesses a sequence of raw texts for NLP tasks 
    using regular expressions and SpaCy.

    This function normalizes an iterable of texts (list or pandas Series) while 
    preserving emotional context. It applies initial cleaning via regular expressions 
    (HTML removal, emoji-to-text conversion, lowercase conversion, character truncation, 
    and punctuation tagging), then passes the precleaned corpus to SpaCy's `nlp.pipe` 
    pipeline for tokenization and lemmatization. Negation particles are retained 
    to preserve sentiment, while standard stop words, punctuation, spaces, URLs, 
    emails, and numbers are filtered out.

    Args:
        texts (list of str or pd.Series): A sequence of raw input text strings to be processed.

    Returns:
        list of str: A list where each element is a string of cleaned and lemmatized 
                     tokens, separated by spaces, corresponding to the input sequence.

    Notes:
        - Requires a globally initialized SpaCy language model named `nlp`.
        - Uses `nlp.pipe` with a batch size of 256.
        - Converts repeated punctuation into custom tokens: 'multiexcl', 
          'multiquest', and 'multiellipsis'.
    """

    # REGULAR EXPRESSIONS 
    
    precleaned_texts = []
    
    # Convert to list if a pandas Series is passed
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
        
    for text in texts:
        text = re.sub(r'<.*?>', ' ', text)                 # Remove HTML tags (e.g., <br>, <div>)
        text = emoji.demojize(text, delimiters=(" ", " ")) # Convert emojis to text (e.g., 😊 -> :smile:)
        text = text.lower()                                # Convert text to lowercase
        text = re.sub(r'!{2,}', ' multiexcl ', text)      # Replace 2 or more consecutive exclamation marks with a special token 
        text = re.sub(r'\?{2,}', ' multiquest ', text)     # Replace 2 or more consecutive question marks with a special token
        text = re.sub(r'\.{3,}', ' multiellipsis ', text)  # Replace 3 or more consecutive periods with an ellipsis token   
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)     # Keep a maximum of 2 consecutive identical letters ("coooool" -> "cool")       
        precleaned_texts.append(text)

    # LEMMATIZATION  
    
    final_results = []
    protected_words = {"not", "no", "n't", "never", "nothing", "neither", "nowhere", "cannot"}
    
    # nlp.pipe processes texts in batches (batch_size)
    for doc in nlp.pipe(precleaned_texts, batch_size=256):
        lemmatized = []
        for token in doc:
            if token.text in protected_words:
                lemmatized.append(token.text)
            elif (
                not token.is_stop          # Remove stop words (the, is, in, at)  
                and not token.is_punct     # Remove punctuation      
                and not token.is_space     # Remove extra spaces and line breaks
                and not token.like_num     # Remove numbers (rarely useful for sentiment analysis)
                and not token.like_url     # Remove URLs (e.g., http://...)
                and not token.like_email   # Remove email addresses
            ):
                lemmatized.append(token.lemma_)
                
        # Join the words back into a string and add to the final list
        final_results.append(' '.join(lemmatized))

    return final_results
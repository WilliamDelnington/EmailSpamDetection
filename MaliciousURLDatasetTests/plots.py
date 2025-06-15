import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Combine "combined_df.csv" and "combined_val_df.csv" into one dataframe
df = pd.concat([pd.read_csv('phishing_features_train.csv'), pd.read_csv('phishing_features_val.csv')], ignore_index=True)

# Define the columns to plot
columns_to_plot = ['redirects', 'not_indexed_by_google', 'issuer', 'certificate_age', 'email_submission', 'request_url_percentage', 'url_anchor_percentage', 'meta_percentage', 'script_percentage', 'link_percentage', 'mouseover_changes', 'right_click_disabled', 'popup_window_has_text_field', 'use_iframe', 'has_suspicious_port', 'external_favicons', 'TTL', 'ip_address_count', 'TXT_record', 'check_sfh', 'count_domain_occurrences', 'domain_registeration_length', 'abnormal_url', 'age_of_domain', 'page_rank_decimal']

# Create a list to store the file names of the saved plots
file_names = []

# Loop through the columns and create the scatterplot or barplot
for column in columns_to_plot:
    if df[column].dtype == 'int64' or df[column].dtype == 'float64':
        fig, ax = plt.subplots()
        sns.regplot(x=column, y='is_malicious', data=df, ax=ax)
        corr_coef = df[[column, 'is_malicious']].corr().iloc[0,1]
        ax.set_title(f'{column} vs is_malicious\nCorrelation Coefficient: {corr_coef:.2f}')
        file_name = f'{column}_scatterplot.png'
        plt.savefig(file_name)
        file_names.append(file_name)
    elif df[column].dtype == 'object':
        fig, ax = plt.subplots()
        if (df[column] == "None").sum() > 0:
            sns.countplot(x=column, hue='is_malicious', data=df[df[column] == "None"], ax=ax)
            ax.set_title(f'{column} (null) vs is_malicious')
            file_name = f'{column}_null_barplot.png'
            plt.savefig(file_name)
            file_names.append(file_name)
        sns.countplot(x=column, hue='is_malicious', data=df, ax=ax)
        ax.set_title(f'{column} (all) vs is_malicious')
        file_name = f'{column}_all_barplot.png'
        plt.savefig(file_name)
        file_names.append(file_name)

# Create a figure with subplots to combine the saved plots
num_plots = len(file_names)
num_rows = int(np.ceil(num_plots/2))
fig, axs = plt.subplots(num_rows, 2, figsize=(20, 5*num_rows))
for i, file_name in enumerate(file_names):
    row = i // 2
    col = i % 2
    img = plt.imread(file_name)
    axs[row, col].imshow(img)
    axs[row, col].axis('off')
if num_plots % 2 == 1:
    axs[num_rows-1, 1].axis('off')
plt.tight_layout()
plt.savefig('correlation_coefficient.png')

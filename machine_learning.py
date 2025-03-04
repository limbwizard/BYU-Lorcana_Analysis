# Data handling and analysis
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, roc_auc_score,
                           accuracy_score, precision_score, 
                           recall_score, f1_score)

# ML models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV

# Visualization
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt

class LorcanaGameAnalysis:
    def __init__(self):
        # DataFrame to store model performance metrics
        self.results_df = pd.DataFrame(columns=[
            'classifier', 'accuracy', 'precision', 'recall', 'f1_score', 'auc',
            'ci_lower', 'ci_upper'
        ])
        
        # DataFrame to store cross-validation results
        self.cv_results_df = pd.DataFrame(columns=[
            'Classifier', 'Cross-validation_Score', 'Cross-validation_Standard_Deviation'
        ])
        
        # Store trained models for ensemble
        self.trained_models = {}
        
        # Lists to store ROC curve data
        self.roc_curves = []
        self.auc_scores = []
        self.classifier_names = []
        
        # Dictionary of models to evaluate
        self.models = {
            'KNN': (KNeighborsClassifier(
                n_neighbors=5,
                weights='distance',
                metric='manhattan'
            ), False),
            'Decision Tree': (DecisionTreeClassifier(
                random_state=42,
                min_samples_leaf=4,
                min_samples_split=10,
                max_depth=5,
                class_weight='balanced'
            ), False),
            'Random Forest': (RandomForestClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                max_features='sqrt',
                max_depth=8,
                class_weight='balanced',
                random_state=42
            ), False),
            'Neural Network': (MLPClassifier(
                hidden_layer_sizes=(32, 16, 8),
                max_iter=2000,
                early_stopping=True,
                learning_rate='adaptive',
                random_state=42
            ), False),
            'Naive Bayes': (GaussianNB(
                var_smoothing=1e-7
            ), False),
            'SVM': (SVC(
                kernel='rbf',
                C=5.0,
                gamma='scale',
                class_weight='balanced',
                random_state=42
            ), True)
        }

    def perform_statistical_analysis(self, df):
        # Overall Statistics
        total_games = len(df)
        overall_win_rate = df['win'].mean()
        print(f"\nSample Size: {total_games} games")
        print(f"Overall Win Rate: {overall_win_rate:.1%}")
        
        # Starting Player Analysis
        first_player_wins = df[df['starting_player'] == 1]['win'].mean()
        second_player_wins = df[df['starting_player'] == 0]['win'].mean()
        
        print("\nStarting Player Effect (1 = first player):")
        print(f"First Player Win Rate:  {first_player_wins:.1%}")
        print(f"Second Player Win Rate: {second_player_wins:.1%}")
        
        # T-test for starting player
        first_player_results = df[df['starting_player'] == 1]['win']
        second_player_results = df[df['starting_player'] == 0]['win']
        t_stat, p_value = stats.ttest_ind(first_player_results, second_player_results)
        
        print(f"\nStatistical significance calculated using two-sample t-test")
        print(f"T-statistic: {t_stat:.3f}")
        print(f"P-value: {p_value:.3f}")
        
        # Card Analysis
        print("\nCard Analysis:")
        print("Win Rate With: Percentage of games won when card was present in deck")
        print("Win Rate Without: Percentage of games won when card was absent from deck")
        print("Positive difference = better win rate when card is included\n")
        
        card_stats = []
        p_values = []
        
        for card in df.columns:
            if card not in ['win', 'game', 'starting_player']:
                card_data = df[df[card] == 1]
                games_with_card = len(card_data)
                if games_with_card == 0:
                    continue
                
                win_rate_with = card_data['win'].mean()
                win_rate_without = df[df[card] == 0]['win'].mean()
                
                with_card = df[df[card] == 1]['win']
                without_card = df[df[card] == 0]['win']
                t_stat, p_value = stats.ttest_ind(with_card, without_card)
                
                p_values.append(p_value)
                
                card_stats.append({
                    'card': card,
                    'win_rate_with': win_rate_with,
                    'win_rate_without': win_rate_without,
                    'games_played': games_with_card,
                    'difference': win_rate_with - win_rate_without,
                    't_stat': t_stat,
                    'p_value': p_value,
                    'significant': False  # Will update after multiple testing correction
                })
        
        # Apply multiple testing correction (Benjamini-Hochberg)
        if p_values:
            adjusted_p = multipletests(p_values, method='fdr_bh')[1]  # FDR correction
            
            # Update significance values with adjusted p-values
            for i, stat in enumerate(card_stats):
                stat['adjusted_p_value'] = adjusted_p[i]
                stat['significant'] = adjusted_p[i] < 0.05
        
        card_stats_df = pd.DataFrame(card_stats)
        
        if not card_stats_df.empty:
            # Create impact comparison table
            sorted_impact = card_stats_df.sort_values('difference', ascending=False)
            top_positive = sorted_impact.head(5)
            top_negative = sorted_impact.tail(5)
            
            # Create separator
            separator = pd.DataFrame({'card': ['...'], 
                                    'win_rate_with': [np.nan],
                                    'win_rate_without': [np.nan],
                                    'difference': [np.nan],
                                    'p_value': [np.nan],
                                    'adjusted_p_value': [np.nan]})
            
            # Combine tables
            impact_table = pd.concat([top_positive, separator, top_negative])[
                ['card', 'win_rate_with', 'win_rate_without', 'difference', 'p_value', 'adjusted_p_value']
            ].reset_index(drop=True)

            print("\nMost Impactful Cards (Presence on Win Rate):")
            print(impact_table.to_string(formatters={
                'win_rate_with': '{:.1%}'.format,
                'win_rate_without': '{:.1%}'.format,
                'difference': '{:+.3f}'.format,
                'p_value': '{:.3f}'.format,
                'adjusted_p_value': '{:.3f}'.format
            }, index=False, na_rep=''))
        
        return card_stats_df

    def load_data(self, file_path):
        # Load data
        df = pd.read_csv(file_path)
        self.card_stats_df = self.perform_statistical_analysis(df)
        
        # Split features and target
        X = df.drop(['win', 'game'], axis=1)
        y = df['win']
        
        # Convert to numpy arrays
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X.values, y.values, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train)
        self.X_test_scaled = scaler.transform(self.X_test)
        
        # Store feature names
        self.feature_names = X.columns.tolist()

    def evaluate_model(self, clf, name, needs_calibration=False):
        print(f"Training {name}...")
        
        # Fit classifier
        clf.fit(self.X_train_scaled, self.y_train)
        
        if needs_calibration:
            clf_proba = CalibratedClassifierCV(estimator=clf, cv=5, method='sigmoid')
            clf_proba.fit(self.X_train_scaled, self.y_train)
            y_pred = clf_proba.predict(self.X_test_scaled)  # FIXED: Use calibrated model for predictions
            y_pred_proba = clf_proba.predict_proba(self.X_test_scaled)[:, 1]
        else:
            y_pred = clf.predict(self.X_test_scaled)
            y_pred_proba = clf.predict_proba(self.X_test_scaled)[:, 1]

        # Calculate metrics
        metrics = {
            'classifier': name,
            'accuracy': round(accuracy_score(self.y_test, y_pred), 3),
            'precision': round(precision_score(self.y_test, y_pred), 3),
            'recall': round(recall_score(self.y_test, y_pred), 3),
            'f1_score': round(f1_score(self.y_test, y_pred), 3),
        }
        
        # Bootstrap confidence intervals - FIXED
        n_iterations = 1000
        scores = []
        for _ in range(n_iterations):
            # Sample with replacement from test indices
            indices = np.random.choice(range(len(self.y_test)), size=len(self.y_test), replace=True)
            # Calculate accuracy on this bootstrap sample
            score = accuracy_score(self.y_test[indices], y_pred[indices])
            scores.append(score)
        confidence_interval = np.percentile(scores, [2.5, 97.5])
        metrics['ci_lower'] = round(confidence_interval[0], 3)
        metrics['ci_upper'] = round(confidence_interval[1], 3)
        
        # Calculate ROC and AUC
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        metrics['auc'] = round(roc_auc_score(self.y_test, y_pred_proba), 3)
        
        self.roc_curves.append((fpr, tpr))
        self.auc_scores.append(metrics['auc'])
        self.classifier_names.append(name)
        
        # Store model for ensemble
        self.trained_models[name] = clf if not needs_calibration else clf_proba
        
        # Cross-validation
        cv_scores = cross_val_score(clf, self.X_train_scaled, self.y_train, cv=10, scoring='accuracy')
        
        # Store results
        self.results_df.loc[len(self.results_df)] = metrics
        self.cv_results_df.loc[len(self.cv_results_df)] = {
            'Classifier': name,
            'Cross-validation_Score': round(cv_scores.mean(), 3),
            'Cross-validation_Standard_Deviation': round(cv_scores.std(), 3)
        }
        
        return clf

    def train_all_models(self):
        for name, (model, needs_calibration) in self.models.items():
            self.evaluate_model(model, name, needs_calibration)

    def plot_roc_curves(self):
        plt.figure(figsize=(10, 8))
        
        for i, (name, (fpr, tpr)) in enumerate(zip(self.classifier_names, self.roc_curves)):
            plt.plot(fpr, tpr, label=f'{name} (AUC = {self.auc_scores[i]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Different Models')
        plt.legend(loc="lower right")
        
        plt.savefig('roc_curves.png', bbox_inches='tight')
        plt.close()
        
        print("\nResults Summary:")
        print(self.results_df)
        print("\nCross-validation Results:")
        print(self.cv_results_df)

    def analyze_feature_importance(self):
        print("\n=== Feature Importance Analysis ===")
        rf_model = self.trained_models['Random Forest']
        
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Get all features and their importance scores
        all_features = [self.feature_names[i] for i in indices]
        all_scores = importances[indices]
        
        # Create full significance table with stars
        print("\nStatistical Significance of All Features:")
        significance_data = []
        for feature in all_features:
            # Flag special features like 'starting_player'
            is_card = feature != 'starting_player'
            
            if is_card and feature in self.card_stats_df['card'].values:
                stats = self.card_stats_df[self.card_stats_df['card'] == feature].iloc[0]
                significance_data.append({
                    'Feature': feature,
                    'T-statistic': stats['t_stat'],
                    'P-value': stats['p_value'],
                    'Adjusted_P-value': stats['adjusted_p_value'] if 'adjusted_p_value' in stats else np.nan,
                    'Significant': stats['significant'],
                    'Is_Card': is_card
                })
            else:
                significance_data.append({
                    'Feature': feature,
                    'T-statistic': np.nan,
                    'P-value': np.nan,
                    'Adjusted_P-value': np.nan,
                    'Significant': False,
                    'Is_Card': is_card
                })
        
        # Create and print full significance table
        sig_df = pd.DataFrame(significance_data)
        
        # Add star ratings
        star_ranges = [
            (0.01, '★★★'),  # 3 stars for p < 0.01
            (0.05, '★★'),    # 2 stars for p < 0.05
            (0.1, '★'),      # 1 star for p < 0.1
            (1.0, '')        # No stars otherwise
        ]
        
        def get_stars(p_value):
            if pd.isna(p_value):
                return ''
            for threshold, stars in star_ranges:
                if p_value <= threshold:
                    return stars
            return ''
        
        # Use adjusted p-values for star ratings when available
        sig_df['Stars'] = sig_df['Adjusted_P-value'].apply(get_stars)
        
        print(sig_df.to_string(formatters={
            'T-statistic': '{:.2f}'.format,
            'P-value': '{:.3f}'.format,
            'Adjusted_P-value': '{:.3f}'.format,
            'Significant': lambda x: 'Yes' if x else 'No'
        }, index=False))
        
        # Create top/bottom 5 impact bar graph
        self.plot_top_bottom_impact(sig_df)

    def plot_top_bottom_impact(self, sig_df):
        # Filter to only include card features (exclude game mechanics like starting_player)
        card_features = sig_df[sig_df['Is_Card']]
        
        if len(card_features) > 0:
            # Get top 5 and bottom 5 features by T-statistic
            top_features = card_features.nlargest(min(5, len(card_features)), 'T-statistic')
            bottom_features = card_features.nsmallest(min(5, len(card_features)), 'T-statistic')
            
            # Combine and sort
            impact_df = pd.concat([top_features, bottom_features]).sort_values('T-statistic', ascending=False)
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            plt.title("Top 5 and Bottom 5 Card Impacts by T-statistic\n"
                    "Positive values indicate cards associated with higher win rates", 
                    fontsize=14, pad=20)
            
            # Create bar colors based on direction
            colors = ['#1f77b4' if x >= 0 else '#d62728' for x in impact_df['T-statistic']]
            
            bars = plt.barh(impact_df['Feature'], impact_df['T-statistic'], color=colors)
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.2f}',
                        ha='left' if width >= 0 else 'right',
                        va='center',
                        fontsize=10,
                        bbox=dict(facecolor='white', alpha=0.8))
            
            # Add significance stars
            for i, (_, row) in enumerate(impact_df.iterrows()):
                if row['Stars']:
                    plt.text(0.05 if row['T-statistic'] >= 0 else -0.05,
                            i,
                            row['Stars'],
                            ha='left' if row['T-statistic'] >= 0 else 'right',
                            va='center',
                            fontsize=14,
                            color='gold')
            
            plt.xlabel('T-statistic (Standardized Effect Size)', fontsize=12)
            plt.ylabel('Card', fontsize=12)
            plt.grid(axis='x', alpha=0.3)
            
            # Add explanatory text
            plt.figtext(0.5, -0.15, 
                    "T-statistics calculated from win rate differences with/without each card\n"
                    "★★★: p < 0.01, ★★: p < 0.05, ★: p < 0.1 (after multiple testing correction)",
                    ha="center", fontsize=10, style='italic')
            
            plt.tight_layout()
            plt.savefig('card_impact.png', bbox_inches='tight')
            plt.close()

    def create_ensemble(self):
        print("\n=== Ensemble Model Analysis ===")
        
        predictions = {}
        for name, model in self.trained_models.items():
            if hasattr(model, 'predict_proba'):
                try:
                    predictions[name] = model.predict_proba(self.X_test_scaled)[:, 1]
                except Exception as e:
                    print(f"Warning: Could not get probability predictions from {name}. Error: {e}")
        
        if not predictions:
            print("Error: No models available for ensemble predictions.")
            return None, None, None, None, None
        
        # Get weights based on AUC scores
        weights = {name: score for name, score in zip(self.classifier_names, self.auc_scores)
                  if name in predictions}
        
        weighted_pred = np.zeros(len(self.y_test))
        weight_sum = 0
        for name, pred in predictions.items():
            weighted_pred += pred * weights[name]
            weight_sum += weights[name]
        
        if weight_sum > 0:
            weighted_pred /= weight_sum
            
            ensemble_pred = (weighted_pred > 0.5).astype(int)
            
            accuracy = accuracy_score(self.y_test, ensemble_pred)
            precision = precision_score(self.y_test, ensemble_pred)
            recall = recall_score(self.y_test, ensemble_pred)
            f1 = f1_score(self.y_test, ensemble_pred)
            auc = roc_auc_score(self.y_test, weighted_pred)
            
            print("\nEnsemble Model Performance:")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Precision: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-score: {f1:.3f}")
            print(f"AUC: {auc:.3f}")
            
            return accuracy, precision, recall, f1, auc
        else:
            print("Error: No valid weights for ensemble prediction.")
            return None, None, None, None, None

def main():
    analysis = LorcanaGameAnalysis()
    analysis.load_data('initial_data.csv')
    analysis.train_all_models()
    analysis.plot_roc_curves()
    analysis.analyze_feature_importance()
    analysis.create_ensemble()

if __name__ == "__main__":
    main()
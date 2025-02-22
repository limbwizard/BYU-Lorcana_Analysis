# Data handling and analysis
import pandas as pd
import numpy as np
from scipy import stats

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
        
        print("\nStarting Player Effect:")
        print(f"First Player Win Rate:  {first_player_wins:.1%}")
        print(f"Second Player Win Rate: {second_player_wins:.1%}")
        
        # T-test for starting player
        first_player_results = df[df['starting_player'] == 1]['win']
        second_player_results = df[df['starting_player'] == 0]['win']
        t_stat, p_value = stats.ttest_ind(first_player_results, second_player_results)
        
        print(f"T-statistic: {t_stat:.3f}")
        print(f"P-value: {p_value:.3f}")
        
        # Card Analysis
        print("\nCard Analysis:")
        card_stats = []
        
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
                
                card_stats.append({
                    'card': card,
                    'win_rate_with': win_rate_with,
                    'win_rate_without': win_rate_without,
                    'games_played': games_with_card,
                    'difference': win_rate_with - win_rate_without,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                })
        
        card_stats_df = pd.DataFrame(card_stats)
        significant_cards = card_stats_df[card_stats_df['significant']].sort_values('difference', ascending=False)
        
        print("\nSignificant Card Effects:")
        if len(significant_cards) > 0:
            print(significant_cards[['card', 'win_rate_with', 'win_rate_without', 
                                   'games_played', 'difference', 'p_value']].to_string(index=False))
        else:
            print("No cards showed statistically significant impact on win rate.")
        
        return card_stats_df

    def load_data(self, file_path):
        # Load data
        df = pd.read_csv(file_path)
        self.perform_statistical_analysis(df)
        
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
            y_pred = clf.predict(self.X_test_scaled)
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
        
        # Bootstrap confidence intervals
        n_iterations = 1000
        scores = []
        for _ in range(n_iterations):
            indices = np.random.randint(0, len(y_pred), len(y_pred))
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
        
        print("\nTop 10 Most Important Features:")
        for f in range(min(10, len(self.feature_names))):
            print("%d. %s (%f)" % (f + 1, self.feature_names[indices[f]], importances[indices[f]]))
        
        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(10), importances[indices[:10]])
        plt.xticks(range(10), [self.feature_names[i] for i in indices[:10]], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.close()

    def create_ensemble(self):
        print("\n=== Ensemble Model Analysis ===")
        
        predictions = {}
        for name, model in self.trained_models.items():
            if hasattr(model, 'predict_proba'):
                predictions[name] = model.predict_proba(self.X_test_scaled)[:, 1]
        
        weights = {name: score for name, score in zip(self.classifier_names, self.auc_scores)}
        
        weighted_pred = np.zeros(len(self.y_test))
        weight_sum = 0
        for name, pred in predictions.items():
            weighted_pred += pred * weights[name]
            weight_sum += weights[name]
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

def main():
    analysis = LorcanaGameAnalysis()
    analysis.load_data('initial_data.csv')
    analysis.train_all_models()
    analysis.plot_roc_curves()
    analysis.analyze_feature_importance()
    analysis.create_ensemble()

if __name__ == "__main__":
    main()
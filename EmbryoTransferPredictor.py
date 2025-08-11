import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import json

import torch, numpy as np, random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class EmbryoTransferNet(nn.Module):
    """Neural network for embryo transfer prediction."""
    
    def __init__(self, input_size=3, hidden_sizes=[64, 32, 16], output_size=2):
        super(EmbryoTransferNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))

        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class EmbryoTransferPredictor:
    def __init__(self, device='cpu'):
        """Initialize the embryo transfer risk predictor."""
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Study outcome data from the research
        self.study_data = {
            'fresh': {
                'live_birth': 168/419,      # 40%
                'clinical_pregnancy': 195/419,  # 47%
                'ongoing_pregnancy': 179/419,   # 43%
                'pregnancy_loss': 20/195,       # 10% of clinical pregnancies
                'ectopic_pregnancy': 7/195      # 4% of clinical pregnancies
            },
            'frozen': {
                'live_birth': 134/419,      # 32%
                'clinical_pregnancy': 164/419,  # 39%
                'ongoing_pregnancy': 149/419,   # 36%
                'pregnancy_loss': 28/164,       # 17% of clinical pregnancies
                'ectopic_pregnancy': 2/164      # 1% of clinical pregnancies
            }
        }
        
        # Initialize model
        self.model = EmbryoTransferNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()
        
        # Feature normalization parameters
        self.feature_means = None
        self.feature_stds = None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
    def calculate_risk_factors(self, age: float, bmi: float, duration: float) -> Dict[str, float]:
        """Calculate individual risk multipliers based on patient characteristics."""
        
        # Age risk (baseline risk = 1.0)
        if age > 37:
            age_risk = 1.6
        elif age > 35:
            age_risk = 1.3
        elif age < 30:
            age_risk = 0.9
        else:
            age_risk = 1.0
            
        # BMI risk (optimal range 20-25)
        if bmi < 18.5:
            bmi_risk = 1.2
        elif 25 <= bmi < 30:
            bmi_risk = 1.1
        elif bmi >= 30:
            bmi_risk = 1.4
        else:
            bmi_risk = 1.0
            
        # Duration of infertility risk
        if duration > 6:
            duration_risk = 1.3
        elif duration > 4:
            duration_risk = 1.2
        else:
            duration_risk = 1.0
            
        return {
            'age_risk': age_risk,
            'bmi_risk': bmi_risk,
            'duration_risk': duration_risk
        }
    
    def adjust_outcomes_for_risk(self, transfer_type: str, risk_multiplier: float) -> Dict[str, float]:
        """Adjust base study outcomes based on individual patient risk."""
        
        base_outcomes = self.study_data[transfer_type]
        
        adjusted = {
            'live_birth': min(base_outcomes['live_birth'] / risk_multiplier, 1.0),
            'pregnancy_loss': min(base_outcomes['pregnancy_loss'] * risk_multiplier, 1.0),
            'ectopic_pregnancy': min(base_outcomes['ectopic_pregnancy'] * risk_multiplier, 1.0)
        }
        
        return adjusted
    
    def calculate_composite_risk_score(self, adjusted_outcomes: Dict[str, float]) -> float:
        """Calculate a weighted composite risk score (lower is better)."""
        
        # Weights based on clinical importance
        weights = {
            'failed_live_birth': 0.6,    # Primary outcome
            'pregnancy_loss': 0.25,      # Secondary safety outcome
            'ectopic_pregnancy': 0.15    # Safety outcome
        }
        
        risk_score = (
            (1 - adjusted_outcomes['live_birth']) * weights['failed_live_birth'] +
            adjusted_outcomes['pregnancy_loss'] * weights['pregnancy_loss'] +
            adjusted_outcomes['ectopic_pregnancy'] * weights['ectopic_pregnancy']
        )
        
        return risk_score
    
    def rule_based_prediction(self, age: float, bmi: float, duration: float) -> Dict:
        """Rule-based prediction using clinical risk factors."""
        
        # Calculate risk factors
        risk_factors = self.calculate_risk_factors(age, bmi, duration)
        total_risk_multiplier = (
            risk_factors['age_risk'] * 
            risk_factors['bmi_risk'] * 
            risk_factors['duration_risk']
        )
        
        # Adjust outcomes for both transfer types
        fresh_adjusted = self.adjust_outcomes_for_risk('fresh', total_risk_multiplier)
        frozen_adjusted = self.adjust_outcomes_for_risk('frozen', total_risk_multiplier)
        
        # Calculate composite risk scores
        fresh_risk_score = self.calculate_composite_risk_score(fresh_adjusted)
        frozen_risk_score = self.calculate_composite_risk_score(frozen_adjusted)
        
        # Determine recommendation
        if fresh_risk_score < frozen_risk_score:
            recommendation = 'fresh'
        else:
            recommendation = 'frozen'
        
        # Calculate confidence based on difference in risk scores
        risk_difference = abs(fresh_risk_score - frozen_risk_score)
        confidence = min(risk_difference * 100, 95)
        
        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'fresh_risk_score': fresh_risk_score,
            'frozen_risk_score': frozen_risk_score,
            'fresh_outcomes': fresh_adjusted,
            'frozen_outcomes': frozen_adjusted,
            'risk_factors': risk_factors,
            'total_risk_multiplier': total_risk_multiplier
        }
    
    def generate_synthetic_data(self, n_samples: int = 5000) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic patient data for model training."""
        
        np.random.seed(42)
        
        # Generate patient characteristics based on study baseline data
        ages = np.random.normal(32.8, 3.9, n_samples)
        bmis = np.random.normal(22.3, 2.65, n_samples)
        durations = np.random.exponential(3.25, n_samples)
        
        # Ensure realistic ranges
        ages = np.clip(ages, 18, 45)
        bmis = np.clip(bmis, 16, 40)
        durations = np.clip(durations, 0.5, 15)
        
        # Create feature matrix
        X = np.column_stack([ages, bmis, durations])
        
        # Generate labels using rule-based system
        y = []
        for i in range(n_samples):
            prediction = self.rule_based_prediction(ages[i], bmis[i], durations[i])
            # 0 = frozen, 1 = fresh
            label = 1 if prediction['recommendation'] == 'fresh' else 0
            y.append(label)
        
        y = np.array(y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        return X_tensor, y_tensor
    
    def normalize_features(self, X: torch.Tensor, fit: bool = False) -> torch.Tensor:
        """Normalize features to zero mean and unit variance."""
        
        if fit:
            self.feature_means = X.mean(dim=0)
            self.feature_stds = X.std(dim=0)
        
        return (X - self.feature_means) / (self.feature_stds + 1e-8)
    
    def train_model(self, n_samples: int = 5000, epochs: int = 200, val_split: float = 0.2):
        """Train the PyTorch neural network."""
        
        print(f"Generating {n_samples} synthetic training samples...")
        X, y = self.generate_synthetic_data(n_samples)
        
        # Split into train/validation
        val_size = int(n_samples * val_split)
        train_size = n_samples - val_size
        
        indices = torch.randperm(n_samples)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Normalize features
        X_train_norm = self.normalize_features(X_train, fit=True)
        X_val_norm = self.normalize_features(X_val, fit=False)
        
        print(f"Training neural network for {epochs} epochs...")
        print(f"Train samples: {train_size}, Validation samples: {val_size}")
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_train_norm)
            loss = self.criterion(outputs, y_train)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Validation
            with torch.no_grad():
                self.model.eval()
                val_outputs = self.model(X_val_norm)
                val_loss = self.criterion(val_outputs, y_val)
                
                # Calculate accuracies
                train_pred = torch.argmax(outputs, dim=1)
                val_pred = torch.argmax(val_outputs, dim=1)
                
                train_acc = (train_pred == y_train).float().mean().item()
                val_acc = (val_pred == y_val).float().mean().item()
                
                self.model.train()
            
            # Store losses
            self.train_losses.append(loss.item())
            self.val_losses.append(val_loss.item())
            
            # Print progress
            if (epoch + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - "
                      f"Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.3f} - "
                      f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.3f}")
        
        print("Training completed!")
        return train_acc, val_acc
    
    def predict_with_neural_network(self, age: float, bmi: float, duration: float) -> Dict:
        """Make prediction using the trained neural network."""
        
        if self.feature_means is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        # Prepare input
        X = torch.FloatTensor([[age, bmi, duration]]).to(self.device)
        X_norm = self.normalize_features(X, fit=False)
        
        # Make prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_norm)
            probabilities = F.softmax(outputs, dim=1).cpu().numpy()[0]  
            prediction = torch.argmax(outputs, dim=1).cpu().numpy()[0]
        
        # Fixed: Define recommendation and confidence variables
        recommendation = 'fresh' if prediction == 1 else 'frozen'
        confidence = max(probabilities) * 100
        
        return {
            'nn_recommendation': recommendation,
            'nn_confidence': confidence,
            'fresh_probability': probabilities[1] * 100,
            'frozen_probability': probabilities[0] * 100
        }
    
    def comprehensive_analysis(self, age: float, bmi: float, duration: float) -> Dict:
        """Perform comprehensive analysis using both approaches."""
        
        # Rule-based prediction
        rule_based = self.rule_based_prediction(age, bmi, duration)
        
        # Neural network prediction
        nn_based = None
        if self.feature_means is not None:
            nn_based = self.predict_with_neural_network(age, bmi, duration)
        
        return {
            'patient_info': {
                'age': age,
                'bmi': bmi,
                'infertility_duration': duration
            },
            'rule_based': rule_based,
            'neural_network': nn_based
        }
    
    def plot_training_history(self):
        """Plot training and validation loss curves."""
        
        if not self.train_losses:
            print("No training history available.")
            return
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss', color='blue', alpha=0.7)
        plt.plot(self.val_losses, label='Validation Loss', color='red', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Smooth the curves for better visualization
        window = 10
        if len(self.train_losses) > window:
            train_smooth = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
            val_smooth = np.convolve(self.val_losses, np.ones(window)/window, mode='valid')
            epochs_smooth = range(window-1, len(self.train_losses))
            plt.plot(epochs_smooth, train_smooth, label='Training Loss (Smoothed)', color='blue')
            plt.plot(epochs_smooth, val_smooth, label='Validation Loss (Smoothed)', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Smoothed Training Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_risk_analysis(self, age: float, bmi: float, duration: float):
        """Create visualizations for the risk analysis."""
        
        analysis = self.comprehensive_analysis(age, bmi, duration)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Risk Score Comparison (Rule-based)
        rule_based = analysis['rule_based']
        risk_scores = [rule_based['fresh_risk_score'], rule_based['frozen_risk_score']]
        colors = ['lightcoral', 'lightblue']
        bars1 = ax1.bar(['Fresh Transfer', 'Frozen Transfer'], risk_scores, color=colors)
        ax1.set_ylabel('Risk Score')
        ax1.set_title('Risk Score Comparison - Rule Based (Lower is Better)')
        ax1.set_ylim(0, max(risk_scores) * 1.2)
        
        # Add value labels
        for bar, score in zip(bars1, risk_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        recommended = rule_based['recommendation']
        ax1.annotate(f'Recommended: {recommended.title()}', 
                    xy=(0.5, 0.95), xycoords='axes fraction', 
                    ha='center', va='top', fontweight='bold', fontsize=12)
        
        # Plot 2: Neural Network Probabilities (if available)
        if analysis['neural_network']:
            nn_data = analysis['neural_network']
            probs = [nn_data['fresh_probability'], nn_data['frozen_probability']]
            bars2 = ax2.bar(['Fresh Transfer', 'Frozen Transfer'], probs, color=colors)
            ax2.set_ylabel('Probability (%)')
            ax2.set_title('Neural Network Predictions')
            ax2.set_ylim(0, 100)
            
            for bar, prob in zip(bars2, probs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{prob:.1f}%', ha='center', va='bottom')
            
            nn_recommended = nn_data['nn_recommendation']
            ax2.annotate(f'NN Recommended: {nn_recommended.title()}', 
                        xy=(0.5, 0.95), xycoords='axes fraction', 
                        ha='center', va='top', fontweight='bold', fontsize=12)
        else:
            ax2.text(0.5, 0.5, 'Neural Network\nNot Trained', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title('Neural Network Predictions')
        
        # Plot 3: Outcome Probabilities
        outcomes = ['Live Birth', 'Pregnancy Loss', 'Ectopic']
        fresh_outcomes = [
            rule_based['fresh_outcomes']['live_birth'] * 100,
            rule_based['fresh_outcomes']['pregnancy_loss'] * 100,
            rule_based['fresh_outcomes']['ectopic_pregnancy'] * 100
        ]
        frozen_outcomes = [
            rule_based['frozen_outcomes']['live_birth'] * 100,
            rule_based['frozen_outcomes']['pregnancy_loss'] * 100,
            rule_based['frozen_outcomes']['ectopic_pregnancy'] * 100
        ]
        
        x = np.arange(len(outcomes))
        width = 0.35
        
        ax3.bar(x - width/2, fresh_outcomes, width, label='Fresh', color='lightcoral', alpha=0.8)
        ax3.bar(x + width/2, frozen_outcomes, width, label='Frozen', color='lightblue', alpha=0.8)
        
        ax3.set_xlabel('Outcomes')
        ax3.set_ylabel('Probability (%)')
        ax3.set_title('Adjusted Outcome Probabilities')
        ax3.set_xticks(x)
        ax3.set_xticklabels(outcomes)
        ax3.legend()
        
        # Plot 4: Risk Factors
        risk_factors = rule_based['risk_factors']
        factors = ['Age Risk', 'BMI Risk', 'Duration Risk']
        values = [risk_factors['age_risk'], risk_factors['bmi_risk'], risk_factors['duration_risk']]
        
        colors_risk = ['red' if v > 1.1 else 'orange' if v > 1.0 else 'green' for v in values]
        bars4 = ax4.bar(factors, values, color=colors_risk, alpha=0.7)
        ax4.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline Risk')
        ax4.set_ylabel('Risk Multiplier')
        ax4.set_title('Individual Risk Factors')
        ax4.set_ylim(0, max(values) * 1.2)
        ax4.legend()
        
        # Add value labels
        for bar, value in zip(bars4, values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return analysis
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_means': self.feature_means,
            'feature_stds': self.feature_stds,
            'study_data': self.study_data
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.feature_means = checkpoint['feature_means']
        self.feature_stds = checkpoint['feature_stds']
        print(f"Model loaded from {filepath}")

def main():
    """Main function to demonstrate the predictor."""
    
    print("=== PyTorch Embryo Transfer Risk Predictor ===\n")
    
    # Initialize predictor
    predictor = EmbryoTransferPredictor()
    
    # Train the neural network
    print("Training neural network model...")
    train_acc, val_acc = predictor.train_model(n_samples=5000, epochs=200)
    print(f"Final Training Accuracy: {train_acc:.3f}")
    print(f"Final Validation Accuracy: {val_acc:.3f}\n")
    
    # Show training history
    predictor.plot_training_history()
    
    # Example patient cases
    test_cases = [
        {"name": "Young, Normal BMI", "age": 28, "bmi": 22, "duration": 2},
        {"name": "Older, Normal BMI", "age": 38, "bmi": 23, "duration": 4},
        {"name": "Normal Age, High BMI", "age": 32, "bmi": 31, "duration": 3},
        {"name": "High Risk Profile", "age": 40, "bmi": 32, "duration": 7}
    ]
    
    print("\n=== Patient Case Analysis ===")
    for case in test_cases:
        print(f"\n--- {case['name']} ---")
        print(f"Age: {case['age']}, BMI: {case['bmi']}, Duration: {case['duration']} years")
        
        analysis = predictor.comprehensive_analysis(case['age'], case['bmi'], case['duration'])
        
        # Rule-based results
        rule_based = analysis['rule_based']
        print(f"Rule-based: {rule_based['recommendation'].title()} (Conf: {rule_based['confidence']:.1f}%)")
        
        # Neural network results
        if analysis['neural_network']:
            nn_based = analysis['neural_network']
            print(f"Neural Net: {nn_based['nn_recommendation'].title()} (Conf: {nn_based['nn_confidence']:.1f}%)")
            
            # Check agreement
            agreement = rule_based['recommendation'] == nn_based['nn_recommendation']
            print(f"Agreement: {'✓' if agreement else '✗'}")
    
    # Interactive mode
    print("\n=== Interactive Analysis ===")
    print("Enter patient details for personalized analysis:")
    
    try:
        age = float(input("Patient age (18-45): "))
        bmi = float(input("Patient BMI (16-40): "))
        duration = float(input("Infertility duration in years (0.5-15): "))
        
        print(f"\nAnalyzing patient: Age {age}, BMI {bmi}, Duration {duration} years")
        
        # Perform comprehensive analysis and create visualizations
        analysis = predictor.plot_risk_analysis(age, bmi, duration)
        
        print("\n=== COMPREHENSIVE RESULTS ===")
        
        # Rule-based results
        rule_based = analysis['rule_based']
        print(f"\nRule-Based Analysis:")
        print(f"  Recommendation: {rule_based['recommendation'].upper()}")
        print(f"  Confidence: {rule_based['confidence']:.1f}%")
        print(f"  Fresh Risk Score: {rule_based['fresh_risk_score']:.3f}")
        print(f"  Frozen Risk Score: {rule_based['frozen_risk_score']:.3f}")
        
        # Neural network results
        if analysis['neural_network']:
            nn_based = analysis['neural_network']
            print(f"\nNeural Network Analysis:")
            print(f"  Recommendation: {nn_based['nn_recommendation'].upper()}")
            print(f"  Confidence: {nn_based['nn_confidence']:.1f}%")
            print(f"  Fresh Probability: {nn_based['fresh_probability']:.1f}%")
            print(f"  Frozen Probability: {nn_based['frozen_probability']:.1f}%")
            
            # Final consensus
            agreement = rule_based['recommendation'] == nn_based['nn_recommendation']
            print(f"\nModel Agreement: {'✓ CONSENSUS' if agreement else '✗ DISAGREEMENT'}")
            
            if agreement:
                print(f"FINAL RECOMMENDATION: {rule_based['recommendation'].upper()} EMBRYO TRANSFER")
            else:
                print("FINAL RECOMMENDATION: Consider both approaches - consult with clinical team")
        
        # Save model option
        save_model = input("\nSave trained model? (y/n): ").lower().strip()
        if save_model == 'y':
            predictor.save_model("embryo_transfer_model.pth")
        
    except (ValueError, KeyboardInterrupt):
        print("\nAnalysis completed.")

if __name__ == "__main__":
    main()

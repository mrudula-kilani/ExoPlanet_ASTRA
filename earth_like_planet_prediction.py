# Earth-like Planet Prediction System using TensorFlow
# For Hackathon - Deep Learning approach to predict Earth-like planets

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("🚀 TensorFlow Earth-like Planet Prediction System")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Step 1: Generate synthetic exoplanet dataset
print("\n📊 Generating synthetic exoplanet dataset...")
n_samples = 10000  # Larger dataset for deep learning

# Create realistic exoplanet features
data = {
    'radius': np.random.lognormal(0, 1, n_samples),  # Earth radii (log-normal distribution)
    'mass': np.random.lognormal(0, 1.2, n_samples),  # Earth masses
    'orbital_period': np.random.lognormal(5, 1.5, n_samples),  # Days (log-normal)
    'distance_from_star': np.random.lognormal(0, 0.8, n_samples),  # AU
    'stellar_magnitude': np.random.normal(10, 3, n_samples),  # Apparent magnitude
    'stellar_temperature': np.random.normal(5200, 1800, n_samples),  # Kelvin
    'stellar_radius': np.random.lognormal(0, 0.5, n_samples),  # Solar radii
    'eccentricity': np.random.beta(0.5, 2, n_samples),  # Orbital eccentricity
    'metallicity': np.random.normal(0, 0.3, n_samples),  # [Fe/H] ratio
    'discovery_year': np.random.randint(1995, 2024, n_samples),  # Discovery year
}

df = pd.DataFrame(data)

# Enhanced Earth-like classification criteria
def calculate_earth_similarity_score(row):
    """
    Calculate a continuous similarity score to Earth (0-1)
    This creates a more nuanced target for neural network training
    """
    scores = []
    
    # Size similarity (radius: 0.5-2.0 optimal, mass: 0.1-5.0 optimal)
    radius_score = np.exp(-((row['radius'] - 1.0) ** 2) / (2 * 0.5**2))
    mass_score = np.exp(-((row['mass'] - 1.0) ** 2) / (2 * 1.5**2))
    scores.extend([radius_score, mass_score])
    
    # Habitable zone (period: 200-500 days optimal)
    period_score = 1.0 if 200 <= row['orbital_period'] <= 500 else \
                   np.exp(-((row['orbital_period'] - 350) ** 2) / (2 * 200**2))
    distance_score = np.exp(-((row['distance_from_star'] - 1.0) ** 2) / (2 * 0.5**2))
    scores.extend([period_score, distance_score])
    
    # Stellar properties (Sun-like star optimal)
    temp_score = np.exp(-((row['stellar_temperature'] - 5778) ** 2) / (2 * 1000**2))
    stellar_radius_score = np.exp(-((row['stellar_radius'] - 1.0) ** 2) / (2 * 0.3**2))
    scores.extend([temp_score, stellar_radius_score])
    
    # Orbital stability
    eccen_score = 1.0 - row['eccentricity']  # Lower eccentricity is better
    metallicity_score = np.exp(-((row['metallicity'] - 0.0) ** 2) / (2 * 0.2**2))
    scores.extend([eccen_score, metallicity_score])
    
    return np.mean(scores)

# Calculate similarity scores
df['earth_similarity'] = df.apply(calculate_earth_similarity_score, axis=1)

# Create binary classification (threshold at 0.6 for Earth-like)
df['is_earth_like'] = (df['earth_similarity'] > 0.6).astype(int)

print(f"Dataset shape: {df.shape}")
print(f"Earth-like planets: {df['is_earth_like'].sum()} ({df['is_earth_like'].mean()*100:.1f}%)")
print(f"Average Earth similarity score: {df['earth_similarity'].mean():.3f}")

# Display dataset info
print("\n📋 Dataset Statistics:")
print(df.describe())

# Step 2: Data Visualization
print("\n📈 Creating visualizations...")

plt.figure(figsize=(20, 15))

# Plot 1: Earth similarity score distribution
plt.subplot(3, 4, 1)
plt.hist(df[df['is_earth_like']==1]['earth_similarity'], alpha=0.7, label='Earth-like', bins=30, color='green')
plt.hist(df[df['is_earth_like']==0]['earth_similarity'], alpha=0.7, label='Non-Earth-like', bins=30, color='red')
plt.xlabel('Earth Similarity Score')
plt.ylabel('Frequency')
plt.title('Earth Similarity Distribution')
plt.legend()

# Plot 2-5: Feature distributions
features_to_plot = ['radius', 'mass', 'orbital_period', 'stellar_temperature']
for i, feature in enumerate(features_to_plot, 2):
    plt.subplot(3, 4, i)
    plt.scatter(df[df['is_earth_like']==1][feature], 
               df[df['is_earth_like']==1]['earth_similarity'], 
               alpha=0.6, label='Earth-like', color='green', s=10)
    plt.scatter(df[df['is_earth_like']==0][feature], 
               df[df['is_earth_like']==0]['earth_similarity'], 
               alpha=0.6, label='Non-Earth-like', color='red', s=10)
    plt.xlabel(feature.replace('_', ' ').title())
    plt.ylabel('Earth Similarity')
    plt.title(f'{feature.replace("_", " ").title()} vs Similarity')
    plt.legend()

# Plot 6: Correlation heatmap
plt.subplot(3, 4, 6)
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, fmt='.2f', cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix')

# Plots 7-8: Additional analysis
plt.subplot(3, 4, 7)
plt.scatter(df['distance_from_star'], df['orbital_period'], 
           c=df['is_earth_like'], cmap='RdYlGn', alpha=0.6, s=20)
plt.xlabel('Distance from Star (AU)')
plt.ylabel('Orbital Period (days)')
plt.title('Orbital Characteristics')
plt.colorbar(label='Earth-like')

plt.subplot(3, 4, 8)
plt.scatter(df['stellar_temperature'], df['stellar_radius'], 
           c=df['is_earth_like'], cmap='RdYlGn', alpha=0.6, s=20)
plt.xlabel('Stellar Temperature (K)')
plt.ylabel('Stellar Radius (Solar radii)')
plt.title('Stellar Properties')
plt.colorbar(label='Earth-like')

plt.tight_layout()
plt.show()

# Step 3: Prepare data for TensorFlow
print("\n🔧 Preparing data for TensorFlow training...")

# Feature selection
feature_columns = ['radius', 'mass', 'orbital_period', 'distance_from_star', 
                  'stellar_magnitude', 'stellar_temperature', 'stellar_radius',
                  'eccentricity', 'metallicity', 'discovery_year']

X = df[feature_columns].values
y = df['is_earth_like'].values
similarity_scores = df['earth_similarity'].values

# Split the data
X_train, X_temp, y_train, y_temp, sim_train, sim_temp = train_test_split(
    X, y, similarity_scores, test_size=0.3, random_state=42, stratify=y)

X_val, X_test, y_val, y_test, sim_val, sim_test = train_test_split(
    X_temp, y_temp, sim_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape}")
print(f"Validation set: {X_val_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")

# Step 4: Build TensorFlow Models
print("\n🤖 Building TensorFlow models...")

# Model 1: Simple Dense Neural Network
def create_simple_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

# Model 2: Deep Neural Network with Regularization
def create_deep_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        
        tf.keras.layers.Dense(16, activation='relu'),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    return model

# Model 3: Multi-output model (classification + regression)
def create_multi_output_model(input_dim):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    # Shared layers
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Classification head
    class_head = tf.keras.layers.Dense(32, activation='relu')(x)
    class_head = tf.keras.layers.Dropout(0.2)(class_head)
    classification_output = tf.keras.layers.Dense(1, activation='sigmoid', name='classification')(class_head)
    
    # Regression head (similarity score)
    reg_head = tf.keras.layers.Dense(32, activation='relu')(x)
    reg_head = tf.keras.layers.Dropout(0.2)(reg_head)
    regression_output = tf.keras.layers.Dense(1, activation='linear', name='regression')(reg_head)
    
    model = tf.keras.Model(inputs=inputs, outputs=[classification_output, regression_output])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss={'classification': 'binary_crossentropy', 'regression': 'mse'},
        loss_weights={'classification': 1.0, 'regression': 0.5},
        metrics={'classification': ['accuracy', 'precision', 'recall'], 
                'regression': ['mae']}
    )
    return model

# Step 5: Train Models
print("\n🏋️ Training models...")

input_dim = X_train_scaled.shape[1]
models = {}

# Train Simple Model
print("\n📚 Training Simple Dense Model...")
simple_model = create_simple_model(input_dim)
print(simple_model.summary())

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=10, min_lr=0.0001)

history_simple = simple_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

models['Simple'] = {'model': simple_model, 'history': history_simple}

# Train Deep Model
print("\n📚 Training Deep Neural Network...")
deep_model = create_deep_model(input_dim)

history_deep = deep_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

models['Deep'] = {'model': deep_model, 'history': history_deep}

# Train Multi-output Model
print("\n📚 Training Multi-output Model...")
multi_model = create_multi_output_model(input_dim)

history_multi = multi_model.fit(
    X_train_scaled, {'classification': y_train, 'regression': sim_train},
    validation_data=(X_val_scaled, {'classification': y_val, 'regression': sim_val}),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

models['Multi-output'] = {'model': multi_model, 'history': history_multi}

# Step 6: Evaluate Models
print("\n📊 Evaluating models on test set...")

results = {}

for name, model_info in models.items():
    model = model_info['model']
    
    if name == 'Multi-output':
        # Multi-output model evaluation
        predictions = model.predict(X_test_scaled)
        y_pred_prob = predictions[0].flatten()
        similarity_pred = predictions[1].flatten()
    else:
        # Single output models
        y_pred_prob = model.predict(X_test_scaled).flatten()
        similarity_pred = None
    
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Calculate metrics
    test_loss = model.evaluate(X_test_scaled, 
                              {'classification': y_test, 'regression': sim_test} if name == 'Multi-output' else y_test, 
                              verbose=0)
    
    accuracy = np.mean(y_pred == y_test)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_prob,
        'test_loss': test_loss
    }
    
    print(f"\n🎯 {name} Model Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {test_loss if isinstance(test_loss, float) else test_loss[0]:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Step 7: Plot Training History
print("\n📈 Plotting training history...")

plt.figure(figsize=(15, 10))

for i, (name, model_info) in enumerate(models.items()):
    history = model_info['history']
    
    # Accuracy plots
    plt.subplot(2, 3, i+1)
    if name == 'Multi-output':
        plt.plot(history.history['classification_accuracy'], label='Train')
        plt.plot(history.history['val_classification_accuracy'], label='Validation')
    else:
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{name} Model - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plots
    plt.subplot(2, 3, i+4)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{name} Model - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

plt.tight_layout()
plt.show()

# Step 8: Select Best Model and Create Prediction Function
best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
best_model = results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {results[best_model_name]['accuracy']:.4f})")

# Save the best model
best_model.save('best_earth_planet_model.h5')
print("💾 Best model saved as 'best_earth_planet_model.h5'")

# Create prediction function
def predict_planet_tensorflow(radius, mass, orbital_period, distance_from_star, 
                            stellar_magnitude, stellar_temperature, stellar_radius,
                            eccentricity, metallicity, discovery_year):
    """
    Predict if a planet is Earth-like using the trained TensorFlow model
    """
    # Create feature array
    features = np.array([[radius, mass, orbital_period, distance_from_star, 
                         stellar_magnitude, stellar_temperature, stellar_radius,
                         eccentricity, metallicity, discovery_year]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Make prediction
    if best_model_name == 'Multi-output':
        predictions = best_model.predict(features_scaled, verbose=0)
        probability = predictions[0][0][0]
        similarity_score = predictions[1][0][0]
    else:
        probability = best_model.predict(features_scaled, verbose=0)[0][0]
        similarity_score = None
    
    prediction = probability > 0.5
    
    result = {
        'is_earth_like': bool(prediction),
        'probability_earth_like': float(probability),
        'confidence': float(max(probability, 1-probability))
    }
    
    if similarity_score is not None:
        result['similarity_score'] = float(max(0, min(1, similarity_score)))
    
    return result

# Step 9: Test the prediction function
print(f"\n🧪 Testing TensorFlow prediction function:")

# Test cases
test_cases = [
    {
        'name': 'Earth-like (Earth)',
        'params': (1.0, 1.0, 365, 1.0, 4.8, 5778, 1.0, 0.02, 0.0, 2020)
    },
    {
        'name': 'Super-Earth (Kepler-452b)',
        'params': (1.6, 5.0, 385, 1.05, 13.4, 5757, 1.11, 0.1, -0.37, 2015)
    },
    {
        'name': 'Gas Giant (Jupiter-like)',
        'params': (11.2, 317.8, 4333, 5.2, 4.8, 5778, 1.0, 0.05, 0.0, 2020)
    },
    {
        'name': 'Hot Jupiter',
        'params': (1.4, 0.69, 3.5, 0.05, 8.5, 6000, 1.2, 0.0, 0.1, 2018)
    }
]

for test_case in test_cases:
    result = predict_planet_tensorflow(*test_case['params'])
    print(f"\n{test_case['name']}:")
    print(f"  Earth-like: {result['is_earth_like']}")
    print(f"  Probability: {result['probability_earth_like']:.3f}")
    print(f"  Confidence: {result['confidence']:.3f}")
    if 'similarity_score' in result:
        print(f"  Similarity Score: {result['similarity_score']:.3f}")

# Step 10: Model Analysis and Feature Importance
print(f"\n🔍 Analyzing model performance...")

# Confusion Matrix for best model
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Earth-like', 'Earth-like'],
            yticklabels=['Non-Earth-like', 'Earth-like'])
plt.title(f'Confusion Matrix - {best_model_name} Model')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# ROC-like analysis
plt.subplot(1, 2, 2)
probabilities = results[best_model_name]['probabilities']
thresholds = np.linspace(0, 1, 100)
accuracies = []

for threshold in thresholds:
    pred_at_threshold = (probabilities > threshold).astype(int)
    acc = np.mean(pred_at_threshold == y_test)
    accuracies.append(acc)

plt.plot(thresholds, accuracies, 'b-', linewidth=2)
plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold')
best_threshold = thresholds[np.argmax(accuracies)]
plt.axvline(x=best_threshold, color='g', linestyle='--', label=f'Optimal Threshold: {best_threshold:.3f}')
plt.xlabel('Prediction Threshold')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Prediction Threshold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\n✅ TensorFlow model training complete!")
print(f"📊 Best model: {best_model_name} with {results[best_model_name]['accuracy']:.1%} accuracy")
print(f"🔧 Use predict_planet_tensorflow() function for predictions")
print(f"📁 Model saved as 'best_earth_planet_model.h5'")
print(f"⚙️ Scaler object available as 'scaler' variable")

# Final summary
print(f"\n📋 TENSORFLOW MODEL SUMMARY:")
print(f"{'='*50}")
print(f"Dataset size: {len(df):,} planets")
print(f"Features used: {len(feature_columns)}")
print(f"Training samples: {len(X_train):,}")
print(f"Test accuracy: {results[best_model_name]['accuracy']:.1%}")
print(f"Earth-like planets in dataset: {df['is_earth_like'].sum():,} ({df['is_earth_like'].mean()*100:.1f}%)")
print(f"Model architecture: {best_model_name}")
print(f"Ready for deployment! 🚀")

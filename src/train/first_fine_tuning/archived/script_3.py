# Now let's create the training loop and integration with generator.py

training_integration = 
class LYLAATrainer:
    """
    Training orchestrator that integrates the hypertuner with generator.py data
    """
    
    def __init__(self, hypertuner: LYLAAHypertuner, generator_output_dir: str):
        self.hypertuner = hypertuner
        self.generator_output_dir = Path(generator_output_dir)
        
    def load_training_data(self) -> Tuple[List[Dict], List[int]]:
        """
        Load ground truth data from generator.py outputs
        """
        training_data = []
        ground_truth_labels = []
        
        labels_dir = self.generator_output_dir / 'labels'
        
        for detailed_file in labels_dir.glob('*_detailed.json'):
            with open(detailed_file, 'r') as f:
                data = json.load(f)
            
            # Extract axis labels with their ground truth classifications
            for label_type in ['scale_labels', 'tick_labels', 'axis_title']:
                labels = data.get(label_type, [])
                for label in labels:
                    if 'xyxy' in label:
                        # Convert to the format expected by our classifier
                        xyxy = label['xyxy']
                        img_width = 800  # Assuming standard width, get from metadata
                        img_height = 600  # Assuming standard height, get from metadata
                        
                        cx = (xyxy[0] + xyxy[2]) / 2
                        cy = (xyxy[1] + xyxy[3]) / 2
                        width = xyxy[2] - xyxy[0]
                        height = xyxy[3] - xyxy[1]
                        
                        features = {
                            'normalized_pos': (cx / img_width, cy / img_height),
                            'relative_size': (width / img_width, height / img_height),
                            'aspect_ratio': width / (height + 1e-6),
                            'area': width * height,
                            'centroid': (cx, cy),
                            'bbox': xyxy,
                            'dimensions': (width, height)
                        }
                        
                        training_data.append(features)
                        
                        # Ground truth mapping
                        if label_type == 'scale_labels':
                            ground_truth_labels.append(0)
                        elif label_type == 'tick_labels':
                            ground_truth_labels.append(1)
                        else:  # axis_title
                            ground_truth_labels.append(2)
        
        return training_data, ground_truth_labels
    
    def train_epoch(self, training_data: List[Dict], ground_truth: List[int]) -> Tuple[float, float]:
        """
        Train for one epoch
        """
        self.hypertuner.optimizer.zero_grad()
        
        # Forward pass
        predictions = []
        for features in training_data:
            pred = self.hypertuner.classify_single_label(features)
            predictions.append(pred)
        
        # Compute loss
        loss = self.hypertuner.compute_loss(predictions, ground_truth)
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        self.hypertuner.optimizer.step()
        
        # Apply constraints
        self.hypertuner.constrain_parameters()
        
        # Compute accuracy
        accuracy = self.hypertuner.accuracy(predictions, ground_truth)
        
        return loss.item(), accuracy
    
    def train(self, epochs: int = 100, patience: int = 10):
        """
        Full training loop with early stopping
        """
        print("Loading training data...")
        training_data, ground_truth = self.load_training_data()
        print(f"Loaded {len(training_data)} training samples")
        
        if len(training_data) == 0:
            print("No training data found! Make sure generator.py has been run.")
            return
        
        best_loss = float('inf')
        patience_counter = 0
        
        print("Starting hyperparameter optimization...")
        for epoch in range(epochs):
            loss, accuracy = self.train_epoch(training_data, ground_truth)
            
            # Store history
            self.hypertuner.history['losses'].append(loss)
            self.hypertuner.history['accuracies'].append(accuracy)
            self.hypertuner.history['parameters'].append(
                self.hypertuner.get_current_params_dict().copy()
            )
            
            # Print progress
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch:3d}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}")
                
                # Print current parameter values
                if epoch % 50 == 0:
                    params = self.hypertuner.get_current_params_dict()
                    print("Current parameters:")
                    for name, value in list(params.items())[:5]:  # Show first 5
                        print(f"  {name}: {value:.4f}")
                    print("  ...")
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        print("Training completed!")
        print(f"Final loss: {loss:.6f}")
        print(f"Final accuracy: {accuracy:.4f}")
        
        return self.hypertuner.get_current_params_dict()

# Usage example:
def run_hypertuning_example():
    """
    Example of how to run the hypertuning system
    """
    # Initialize hypertuner
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hypertuner = LYLAAHypertuner(device=device)
    
    # Initialize trainer with generator output directory
    trainer = LYLAATrainer(hypertuner, "test_generation")  # Adjust path as needed
    
    # Run training
    optimal_params = trainer.train(epochs=200, patience=20)
    
    # Save results
    results = {
        'optimal_parameters': optimal_params,
        'training_history': hypertuner.history
    }
    
    with open('lylaa_hypertuning_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\\nOptimal parameters saved to 'lylaa_hypertuning_results.json'")
    return optimal_params


print("TRAINING INTEGRATION CREATED")
print("Key Components:")
print("✓ Data loader for generator.py outputs")
print("✓ Complete training loop with backpropagation")
print("✓ Early stopping for convergence")
print("✓ Parameter constraint enforcement")
print("✓ Training history tracking")
print("✓ Automatic result saving")
print()

print("INTEGRATION WORKFLOW:")
print("1. generator.py creates training data with ground truth labels")
print("2. LYLAATrainer loads this data and extracts features")
print("3. LYLAAHypertuner performs differentiable classification")
print("4. Cross-entropy loss is computed against ground truth")
print("5. Gradients are backpropagated to update parameters")
print("6. Process repeats until convergence")
print("7. Optimal parameters are saved for production use")
import re
import pandas as pd
import ast
import spacy
from textblob import TextBlob
from sklearn.metrics import accuracy_score, classification_report

class AdvancedNLPEvaluator:
    def __init__(self, model_name='en_core_web_trf'):
        """
        Initialize with a more comprehensive NLP pipeline
        Using a larger transformer-based model for better accuracy
        """
        self.nlp = spacy.load(model_name)
    
    def preprocess_comment(self, comment):
        """
        Enhanced preprocessing to handle more complex text cleaning
        """
        # Convert to lowercase
        comment = comment.lower()
        
        # Remove special characters and extra whitespaces
        comment = re.sub(r'[^a-zA-Z0-9\s]', '', comment)
        comment = re.sub(r'\s+', ' ', comment).strip()
        
        return comment
    
    def load_golden_data(self, file_path):
        """
        Load golden data with more robust error handling
        """
        try:
            df = pd.read_csv(file_path)
            golden_data = []
            
            for _, row in df.iterrows():
                try:
                    # Safely evaluate POS and dependencies
                    gold_pos = ast.literal_eval(row['gold_pos']) if pd.notna(row['gold_pos']) else []
                    gold_dependencies = ast.literal_eval(row['gold_dependencies']) if pd.notna(row['gold_dependencies']) else []
                    
                    golden_data.append({
                        "sentence": row['sentence'],
                        "gold_pos": gold_pos,
                        "gold_dependencies": gold_dependencies
                    })
                except (SyntaxError, ValueError) as e:
                    print(f"Error processing row: {e}")
            
            return golden_data
        except Exception as e:
            print(f"Error loading golden data: {e}")
            return []
    
    def align_gold_pos(self, gold_pos, tokens):
        """
        More robust alignment of POS tags
        """
        aligned_pos = []
        for i, token in enumerate(tokens):
            if i < len(gold_pos):
                aligned_pos.append((token.text, gold_pos[i][1]))
            else:
                aligned_pos.append((token.text, 'UNKNOWN'))
        return aligned_pos
    
    def evaluate_accuracy(self, golden_data):
        """
        Comprehensive accuracy evaluation with more detailed metrics
        """
        all_gold_pos_tags = []
        all_predicted_pos_tags = []
        
        total_pos_tags = 0
        correct_pos_tags = 0
        total_dependencies = 0
        correct_uas = 0
        correct_las = 0
        
        for data in golden_data:
            # Preprocess sentence
            sentence = self.preprocess_comment(data["sentence"])
            doc = self.nlp(sentence)
            
            # POS Tagging
            predicted_pos = [(token.text, token.tag_) for token in doc]
            gold_pos = self.align_gold_pos(data["gold_pos"], doc)
            
            # Collect tags for comprehensive evaluation
            all_gold_pos_tags.extend([tag for _, tag in gold_pos])
            all_predicted_pos_tags.extend([tag for _, tag in predicted_pos])
            
            # Calculate POS accuracy
            total_pos_tags += len(gold_pos)
            correct_pos_tags += sum(1 for i in range(len(gold_pos)) if gold_pos[i] == predicted_pos[i])
            
            # Dependency Parsing
            predicted_dependencies = [(token.head.text, token.text, token.dep_) for token in doc]
            gold_dependencies = data["gold_dependencies"]
            
            total_dependencies += len(gold_dependencies)
            
            for gold_dep in gold_dependencies:
                if gold_dep[:2] in [(dep[0], dep[1]) for dep in predicted_dependencies]:
                    correct_uas += 1
                if gold_dep in predicted_dependencies:
                    correct_las += 1
        
        # Calculate metrics
        pos_accuracy = correct_pos_tags / total_pos_tags if total_pos_tags > 0 else 0
        uas = correct_uas / total_dependencies if total_dependencies > 0 else 0
        las = correct_las / total_dependencies if total_dependencies > 0 else 0
        
        # Detailed POS tagging report
        pos_classification_report = classification_report(
            all_gold_pos_tags, 
            all_predicted_pos_tags
        )
        
        return {
            'pos_accuracy': pos_accuracy,
            'uas': uas,
            'las': las,
            'pos_classification_report': pos_classification_report
        }

def main():
    # File path to your golden data
    file_path = "generated_golden_data.csv"
    
    # Initialize evaluator
    evaluator = AdvancedNLPEvaluator()
    
    # Load golden data
    golden_data = evaluator.load_golden_data(file_path)
    
    # Evaluate accuracy
    results = evaluator.evaluate_accuracy(golden_data)
    
    # Print results
    print("Accuracy Metrics:")
    print(f"POS Accuracy: {results['pos_accuracy']:.2f}")
    print(f"UAS: {results['uas']:.2f}")
    print(f"LAS: {results['las']:.2f}")
    
    print("\nPOS Classification Report:")
    print(results['pos_classification_report'])

if __name__ == "__main__":
    main()
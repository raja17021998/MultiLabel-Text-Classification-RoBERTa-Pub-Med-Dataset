import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

# model architecture
class RoBERTaPubMed(torch.nn.Module):
    def __init__(self, target_classes=14):
        super(RoBERTaPubMed, self).__init__()
        self.roberta_model = RobertaModel.from_pretrained('roberta-base', return_dict=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(768, target_classes)

    def forward(self, input_ids, attn_mask):
        output = self.roberta_model(
            input_ids=input_ids, 
            attention_mask=attn_mask
        )
        output_dropout = self.dropout(output.pooler_output)
        output = self.linear(output_dropout)
        return output

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Load the saved model
model_path = "best_model.pt" 
model = RoBERTaPubMed()  
print(model.load_state_dict(torch.load(model_path)))
model.eval()

# Load the CSV file 
csv_file = "Multi-Label Text Classification Dataset.csv"  # Path to your CSV file
df = pd.read_csv(csv_file)


k = 5  

df_subset = df.sample(k)


def predict_labels(input_text):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )
    
    # Perform inference
    with torch.no_grad():
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
    
    # Convert logits to binary labels using thresholding
    predicted_labels = (outputs > 0.5).int().tolist()[0]  # Assuming threshold is 0.5
    
    return predicted_labels


for index, row in tqdm(df_subset.iterrows(), total=k):
    input_text = row['Title'] + ". " + row['abstractText']  
    actual_labels = row[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']].tolist()  # Actual labels
    
    # Perform inference
    predicted_labels = predict_labels(input_text)
    
    print(f"\nActual Labels:", actual_labels)
    print("Predicted Labels:", predicted_labels)
    print()

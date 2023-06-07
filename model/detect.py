from torch import nn
import torch
from argparse import Namespace
from transformers import AutoModel, AutoTokenizer

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("silk-road/luotuo-bert")

# Load the pretrained BERT model
model_args = Namespace(do_mlm=None, pooler_type="cls", temp=0.05, mlp_only_train=False, init_embeddings_model=None)
model = AutoModel.from_pretrained("silk-road/luotuo-bert", trust_remote_code=True, model_args=model_args)

# Define your custom Classifier again
class Classifier(nn.Module):
    def __init__(self, bert_model, hidden_size=1536, num_classes=2):
        super(Classifier, self).__init__()
        self.bert_model = bert_model
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, **inputs):
        outputs = self.bert_model(output_hidden_states=True, return_dict=True, sent_emb=True, **inputs)
        pooled_output = outputs[1]
        x = self.linear1(pooled_output)
        x = self.relu(x)
        x = self.linear2(x)
        return x

# Initialize the classifier using the BERT model
classifier = Classifier(model)

# Load the state dict previously saved
classifier.load_state_dict(torch.load('classifier.pth', map_location=device))

# Move the model to the appropriate device
classifier = classifier.to(device)

# Set the model to evaluation mode
classifier.eval()

label_map = {0: "不过审", 1: "过审"}

def predict_text(text, model):
    encoding = tokenizer.encode_plus(
        text,
        max_length=512,
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=True,
        truncation=True,
        return_tensors='pt',
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    _, preds = torch.max(outputs, dim=1)

    # Use the label mapping to convert the predicted label
    pred_label = label_map[preds.item()]

    return pred_label

# Test cases

# text = "胡锡进最近被粉红围攻了，算是被民粹主义反噬了。胡锡进作为民粹主义的头头之一，为中国民粹化也做出突出贡献。不知现在的胡锡进，面对民粹主义狂潮，内心是什么感想？"
# pred_label = predict_text(text, classifier)

# print("Predicted label:", pred_label)

text = "祖国是我们心中的灯塔，照亮我们前进的步伐；祖国是我们自信的源头，赋予我们无穷的力量。"
pred_label = predict_text(text, classifier)

print(text)
print("AI模型审核结果: ", pred_label)

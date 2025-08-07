import torch
import torch.nn as nn
import pandas as pd 
import numpy as np
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset




def data_initialize():
    data = pd.read_csv('IMDB Dataset.csv')

    word_counts = Counter()
    word_to_num={}
    tokenized_reviews=[]

    for review in data['review']:
        review = review.lower()
        review = ''.join(char for char in review if char.isalpha() or char.isspace() )
        words = review.split()
        word_counts.update(words)



    most_common= word_counts.most_common(5000)


    for i, (word,_) in  enumerate(most_common):
            word_to_num[word]=i + 2 

    word_to_num['<PAD>'] = 0  # padding token
    word_to_num['<UNK>'] = 1  # unknown token


    for review in data['review']:
        temp=[]
        review = review.lower()
        review = ''.join(char for char in review if char.isalpha() or char.isspace() )
        review = review.split()

        review = review[:500]

        for word in review:
            temp.append(word_to_num.get(word, 1)) 

        tokenized_reviews.append(temp)


    vocab_size = len(word_to_num)

    review_tensors = [torch.tensor(r, dtype=torch.long) for r in tokenized_reviews]
    tensor_input_train = pad_sequence(review_tensors, batch_first=True, padding_value=0)

    labels = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
    labels_tensor = torch.tensor(labels.values, dtype=torch.long)

    return vocab_size, tokenized_reviews, tensor_input_train, labels_tensor






class RNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,100)
        self.lstm = nn.LSTM(100,128,batch_first=True)
        self.linear = nn.Linear(128,1)
        
        

    def forward(self, tensor_input):
        embedded_input = self.embedding(tensor_input)
        lstm_output, (hidden_state, cell_state)=self.lstm(embedded_input)
        pooled = lstm_output.mean(dim=1)
        out = self.linear(pooled)
        
        return out



    def calculate_error(self):
        pass




def training(myrnn, tokenized_reviews, tensor_input_train,labels_tensor,num_epochs):
    tensor_input_train = tensor_input_train[:4000]
    labels_tensor = labels_tensor[:4000]

    train_dataset = TensorDataset(tensor_input_train, labels_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    
    Optimizer = optim.Adam(myrnn.parameters(),lr=0.001)
    Criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        for batch_data,batch_labels in train_loader:
            Optimizer.zero_grad()
            outputs = myrnn(batch_data).squeeze()
            loss = Criterion(outputs, batch_labels.float())
            loss.backward()
            Optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    torch.save(myrnn.state_dict(), 'model.pth')
    print("Model saved as model.pth")

    

    
def testing(model, tensor_input_train, labels_tensor):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    # Load the original data to get sentences
    data = pd.read_csv('IMDB Dataset.csv')
    
    model.load_state_dict(torch.load('model.pth'))
    print("Model loaded from model.pth")
    
    # Use last 1000 samples as test data
    test_input = tensor_input_train[4000:5000] 
    test_labels = labels_tensor[4000:5000]
    
    model.eval()
    
    with torch.no_grad():
        outputs = model(test_input).squeeze()
        predictions = (torch.sigmoid(outputs) > 0.5).float()
        accuracy = (predictions == test_labels.float()).sum().item() / len(test_labels)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(test_labels.numpy(), predictions.numpy())
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Show 3 examples
    print("\n=== EXAMPLES ===")
    for i in range(3):
        sentence = data.iloc[4000 + i]['review'][:200] + "..."
        actual = "Positive" if test_labels[i] == 1 else "Negative"
        guess = "Positive" if predictions[i] == 1 else "Negative"
        prob = torch.sigmoid(outputs[i]).item()
        
        print(f"Example {i+1}:")
        print(f"  Review: {sentence}")
        print(f"  Actual: {actual}")
        print(f"  Guess: {guess} ({prob:.3f})")
        print()
    
    return accuracy

def main():
    epochs=12
    vocab_size, tokenized_reviews, tensor_input_train,labels_tensor = data_initialize()
    myrnn=RNN(vocab_size)
    # choice = input("choose train/test:")
    # if choice=="train":
    #     training(myrnn, tokenized_reviews, tensor_input_train,labels_tensor,epochs)
    # elif choice=="test":
    #     testing(myrnn,tensor_input_train,labels_tensor)
    # else:
    #     print("invalid choice")

    # training(myrnn, tokenized_reviews, tensor_input_train,labels_tensor,epochs)
    testing(myrnn,tensor_input_train,labels_tensor)

main()

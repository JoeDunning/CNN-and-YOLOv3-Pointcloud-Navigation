# Library Imports ---------------------------------------------------------------------------------------------------------------------

import torch
import torch.optim as optim
import local_dataset as ld

# Training Framework ---------------------------------------------------------------------------------------------------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in ld.train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        running_loss+=loss.item()

    # - Training results
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(ld.train_loader):.8f}')

    ## -- Model eval/validation
    model.eval()
    pos_result = 0
    total_result = 0
    with torch.no_grad():
        for images, labels in ld.val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total_result+=labels.size(0)
            pos_result+=(predicted==labels).sum().item()

    # - Validation results
    print(f'Validation accuracy: {100*pos_result/total_result:.8f}%')
    save_checkpoint(model, optimizer, filename=filename)
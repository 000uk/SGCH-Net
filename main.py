df = pd.read_csv(CSV_PATH)
num_classes = len(df['label'].unique())
num_frames = int(df["frames"].iloc[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = STGCN_Model(num_classes=5).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(total_steps * 0.1), # WARMUP_RATIO), 원래 linear 일때 썻던 값은데 0.1이었거든? 0.05로 줄이래
    num_training_steps=total_steps,
)
criterion = nn.CrossEntropyLoss()

for epoch in tqdm(range(EPOCHS)):
    model.train()
    train_loss = 0
    total = 0
    correct = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = 100. * correct / total
    train_loss /= len(train_loader)

    # --------------------
    # ⭐ Validation step
    # --------------------
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
        
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            val_total += targets.size(0)
            val_correct += predicted.eq(targets).sum().item()

    val_acc = 100. * val_correct / val_total
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        f"Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
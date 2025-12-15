# model = x3d_s(pretrained=True)
# for p in model.parameters():
#     p.requires_grad = False
# for p in model.blocks[-1].proj.parameters():
#     p.requires_grad = True
# model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)
from model.CGCNN_run import FineTune

cgcnn_run = FineTune(root_dir="../../dataset/cifs/",
        save_dir="./pth/5/",
        log_every_n_steps=10,
        eval_every_n_epochs=1,
        epoch=500,
        val_ratio=0.2,
        test_ratio=0.2,
        opti="Adam",
        lr=0.0001,
        momentum=0.9,
        weight_decay=1e-3,
        dataset_file="../../dataset/results/manual_Tom.json",
        batch_size=128,
        n_conv=2,
        atom_fea_len=64,
        h_fea_len=64,
        n_h=1,
        n_out=1,
        random_seed=1226,
        pin_memory=False,
        num_workers=0)
cgcnn_run.train()
loss, acc, f1, recall, precision = cgcnn_run.test()
cgcnn_run.predict()

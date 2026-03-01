import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from TADA_GA import load_multi_source_data, FeatureMapperResNet, Classifier, calculate_metrics

def run_tada_experiment(source_folder, target_name, output_csv=None, repeat=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    target_file = os.path.join(source_folder, f"{target_name}.arff")
    all_metrics = []

    for r in range(repeat):
        source_data_list, source_labels_list, Xt_train_np, yt_train_np, Xt_test_np, yt_test_np, selected_methods = \
            load_multi_source_data(
                source_folders=[source_folder],
                target_file=target_file,
                target_pos_ratio=0.5
            )

        if len(source_data_list) == 0:
            print("❌ Error: No valid source data loaded.")
            return

        Xt_train = torch.from_numpy(Xt_train_np).float().to(device)
        yt_train = torch.from_numpy(yt_train_np).long().to(device)
        Xt_test = torch.from_numpy(Xt_test_np).float().to(device)
        yt_test = torch.from_numpy(yt_test_np).long().to(device)

        mapped_dim = 128
        feature_mappers = [FeatureMapperResNet(input_dim=Xs.shape[1], mapped_dim=mapped_dim).to(device)
                           for Xs in source_data_list]
        target_mapper = FeatureMapperResNet(input_dim=Xt_train.shape[1], mapped_dim=mapped_dim).to(device)
        classifier = Classifier(input_dim=mapped_dim).to(device)

        params = list(classifier.parameters()) + list(target_mapper.parameters())
        for mapper in feature_mappers:
            params += list(mapper.parameters())
        mu = torch.nn.Parameter(torch.tensor(0.5, device=device))
        params += [mu]

        optimizer = torch.optim.Adam(params, lr=1e-3)
        class_weight = torch.ones(2, device=device)

        for epoch in range(50):
            classifier.train()
            target_mapper.train()
            for mapper in feature_mappers:
                mapper.train()

            optimizer.zero_grad()

            source_z_list, source_y_list = [], []
            for mapper, (Xs_np, ys_np) in zip(feature_mappers, zip(source_data_list, source_labels_list)):
                Xs = torch.from_numpy(Xs_np).float().to(device)
                ys = torch.from_numpy(ys_np).long().to(device)
                zs = mapper(Xs)
                source_z_list.append(zs)
                source_y_list.append(ys)

            zt_train = target_mapper(Xt_train)
            zs_all = torch.cat(source_z_list + [zt_train], dim=0)
            ys_all = torch.cat(source_y_list + [yt_train], dim=0)

            preds = classifier(zs_all)
            mmd_loss = ((torch.cat(source_z_list, dim=0).mean(0) - zt_train.mean(0)) ** 2).sum()
            ce_loss = F.cross_entropy(preds, ys_all, weight=F.softmax(class_weight, dim=0))
            total_loss = torch.sigmoid(mu) * mmd_loss + (1 - torch.sigmoid(mu)) * ce_loss

            total_loss.backward()
            optimizer.step()

        classifier.eval()
        target_mapper.eval()
        with torch.no_grad():
            zt_test = target_mapper(Xt_test)
            logits = classifier(zt_test)
            probas = F.softmax(logits, dim=1)[:, 1]
            pred_labels = torch.argmax(logits, dim=1)

        metrics = calculate_metrics(yt_test.cpu().numpy(), pred_labels.cpu().numpy(), probas.cpu().numpy())
        metrics['Augmentation'] = selected_methods[0] if selected_methods else 'none'
        metrics['Target'] = target_file
        metrics['Repeat'] = r + 1
        all_metrics.append(metrics)

        print(f"[Repeat {r+1}/{repeat}] TADA Result:", metrics)

    # 平均结果
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0] if isinstance(all_metrics[0][key], (float, int))}
    avg_metrics['Augmentation'] = all_metrics[0]['Augmentation']
    avg_metrics['Target'] = all_metrics[0]['Target']
    avg_metrics['Repeat'] = repeat

    print("\n✅ TADA Average Result:", avg_metrics)

    if output_csv:
        df = pd.DataFrame([avg_metrics])
        df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
        print(f"✅ TADA average result saved to {output_csv}")

        raw_csv = output_csv.replace(".csv", "_raw.csv")
        df_raw = pd.DataFrame(all_metrics)
        df_raw.to_csv(raw_csv, index=False)
        print(f"📄 Raw TADA repeated results saved to {raw_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_folder', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='tada_result.csv')
    parser.add_argument('--repeat', type=int, default=30)
    args = parser.parse_args()

    run_tada_experiment(
        source_folder=args.source_folder,
        target_name=args.target,
        output_csv=args.output_csv,
        repeat=args.repeat
    )

import argparse
import data_utils
from data_utils import collate_fn, attention_stats
from torch.utils.data import DataLoader
import model
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Examples:")
    parser.add_argument('--input', '-i', type=str, default='data/training_dataset', help='the input folder')
    parser.add_argument('--curated', '-c', type=bool, default=False)
    parser.add_argument('--label_name','-ln',type=str,default="CorAs")
    parser.add_argument('--output', '-o', type=str, help='the output folder', default='data/output')
    parser.add_argument('--test_size', '-ts', type=float, help='the train-test split ratio', default=0.1)
    parser.add_argument('--random_seed', '-rs', type=int, default=0)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--embedding_dim', '-ed', type=int, default=32)
    parser.add_argument('--mlp_dim', '-md', type=int, default=32)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_head', type=int, help='transformer head number', default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_epoch', '-ne', type=int, help='the number of training epochs', default=10)
    parser.add_argument('--model_output', '-mo', type=str, default='model')
    parser.add_argument('--n_layer', type=int, help='the number of encoder layers', default=2)
    parser.add_argument('--layer_id', type=int, help='the id for attention layer to probe', default=1)
    parser.add_argument('--test_only',type=bool, help='only do test with trained models', default=False)
    # parser.add_argument('--max_len', '-ml', type=int, help='max_length', default=32)
    args = parser.parse_args()

    seq_data = data_utils.SeqDataset(args.input, args.curated, args.label_name)
    num_labels = len(seq_data.label_dict)
    max_len = seq_data.max_len
    train_seq_data, valid_seq_data, test_seq_data = data_utils.data_split(seq_data, args.test_size, args.random_seed)
    train_sampler = data_utils.LengthSortSampler(train_seq_data)
    valid_sampler = data_utils.LengthSortSampler(valid_seq_data)
    test_sampler = data_utils.LengthSortSampler(test_seq_data)
    # train_data_loader = DataLoader(train_seq_data, batch_size=args.batch_size,
    #                                sampler=train_sampler, collate_fn=collate_fn)
    # valid_data_loader = DataLoader(valid_seq_data, batch_size=args.batch_size,
    #                                sampler=valid_sampler, collate_fn=collate_fn)
    # test_data_loader = DataLoader(test_seq_data, batch_size=args.batch_size,
    #                               sampler=test_sampler, collate_fn=collate_fn)

    train_data_loader = DataLoader(train_seq_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_data_loader = DataLoader(valid_seq_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_data_loader = DataLoader(test_seq_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    if not args.test_only:
        seq_model = model.SeqPredictModel(embedding_dim=args.embedding_dim, num_heads=args.n_head, mlp_dim=args.mlp_dim,
                                          dropout_rate=args.dropout, max_len=max_len, n_class=num_labels,
                                          n_layer=args.n_layer, device=args.device)
        if args.device != "cpu":
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        seq_model.to(seq_model.device)
        seq_model.model_training(train_data_loader, valid_data_loader, args.n_epoch, args.model_output)
    # torch.save(seq_model, args.model_output + "/trained_model.pt")
    print('Loading trained model')
    load_model = torch.load(args.model_output + "/trained_model.pt")
    print(load_model.device)
    pred_res, true_label, attention_weights = load_model.model_test(test_data_loader)
    test_seq = []
    for i in test_seq_data.indices:
        test_seq.append(seq_data[i])
    attn_pair = attention_stats(attention_weights, test_seq, seq_data.id_2_token)
    sorted_pairs = sorted(attn_pair.items(), key=lambda x: x[1], reverse=True)
    print(sorted_pairs)
    # data_utils.plot_cm(pred_res, true_label)
    # pred_res, true_label, attention_weights = load_model.model_test(train_data_loader)
    # test_seq = []
    # for i in train_data_loader.sampler.data_source.indices:
    #     test_seq.append(seq_data[i])
    # attn_pair = attention_stats(attention_weights, test_seq, seq_data.id_2_token)
    # sorted_pairs = sorted(attn_pair.items(), key=lambda x: x[1], reverse=True)
    # print(sorted_pairs)
    # data_utils.plot_cm(pred_res, true_label)

import  config

def model_build(args, model_file, dataset_train, dataset_val):
    if args.run_script == 'Training':
        dataset = dataset_train
    else:
        dataset = dataset_val

    if args.network_type == 'bbox_detection':
        legonet = model_file.BBOX_Detection(num_classes=dataset.num_classes())

    elif args.network_type == 'both' or args.network_type == "both_for_roots_2":
        legonet = model_file.PerObjectEstimate(dataset = dataset, network_type = args.network_type,
                                num_classes=dataset.num_classes(),
                                freeze_detection = args.freeze_detection
                                )


    return legonet
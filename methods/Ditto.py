import yaml
import json

from entities.Ditto import DittoClient, DittoServer
from utils.drift import sudden_drift, incremental_drift
from utils.gen_dataset import distribute_dataset

if __name__ == "__main__":
    # load configuration
    with open("../configs/Ditto.yaml", 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        print(json.dumps(args, indent=4))

    # initialize clients and server
    client_train_set, client_test_set, global_test_sets = distribute_dataset(
        args["dataset"], args["client_num"], args["partition"], args["alpha"], args["seed"]
    )

    clients = []
    for client_id in range(args["client_num"]):
        client = DittoClient(client_id, args, client_train_set[client_id], client_test_set[client_id], 0)
        clients.append(client)

    server = DittoServer(args)
    server.get_client_data_size(clients)

    for _round in range(args["rounds"]):
        if args["drift_pattern"] == "sudden" and _round == 100:
            sudden_drift(clients, global_test_sets, _round)
        elif args["drift_pattern"] == "recurrent" and _round in [100, 150]:
            sudden_drift(clients, global_test_sets, _round)
        elif args["drift_pattern"] == "incremental" and _round in [100, 110, 120]:
            incremental_drift(clients, global_test_sets, _round)

        selected_clients = server.select_clients(clients)
        server.send_params(selected_clients)

        for client in selected_clients:
            client.train()
            client.personalized_train(server.model)

        server.aggregate_by_updates(selected_clients)

        # evaluate selected clients
        # if _round % 1 == 0:
        #     local_accuracy = server.local_evaluate(selected_clients, _round)
        #     global_accuracy = server.global_evaluate(selected_clients, global_test_sets, _round)
        #     print(f"Round {_round} | Local accuracy: {local_accuracy} | Global accuracy: {global_accuracy}")

    server.last_round_evaluate(clients, global_test_sets)

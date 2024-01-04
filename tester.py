import torch

from measure import Measure


class Tester:
    def __init__(self, dataset, model, valid_or_test, model_name):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.eval()
        self.dataset = dataset
        self.ary_list = self.dataset.arity_lst
        self.model_name = model_name
        self.valid_or_test = valid_or_test
        self.measure = Measure()
        self.all_facts_as_set_of_tuples = set(self.allFactsAsTuples(self.ary_list))

    def get_rank(self, sim_scores):
        # Assumes the test fact is the first one
        return (sim_scores >= sim_scores[0]).sum()

    def create_queries(self, fact, position):
        if len(fact) == 3:
            r, e1, e2 = fact
            if position == 1:
                return [(r, i, e2) for i in range(1, self.dataset.num_ent)]
            elif position == 2:
                return [(r, e1, i) for i in range(1, self.dataset.num_ent)]
        if len(fact) == 4:
            r, e1, e2, e3 = fact
            if position == 1:
                return [(r, i, e2, e3) for i in range(1, self.dataset.num_ent)]
            elif position == 2:
                return [(r, e1, i, e3) for i in range(1, self.dataset.num_ent)]
            elif position == 3:
                return [(r, e1, e2, i) for i in range(1, self.dataset.num_ent)]
        if len(fact) == 5:
            r, e1, e2, e3, e4 = fact
            if position == 1:
                return [(r, i, e2, e3, e4) for i in range(1, self.dataset.num_ent)]
            elif position == 2:
                return [(r, e1, i, e3, e4) for i in range(1, self.dataset.num_ent)]
            elif position == 3:
                return [(r, e1, e2, i, e4) for i in range(1, self.dataset.num_ent)]
            elif position == 4:
                return [(r, e1, e2, e3, i) for i in range(1, self.dataset.num_ent)]
        if len(fact) == 6:
            r, e1, e2, e3, e4, e5 = fact
            if position == 1:
                return [(r, i, e2, e3, e4, e5) for i in range(1, self.dataset.num_ent)]
            elif position == 2:
                return [(r, e1, i, e3, e4, e5) for i in range(1, self.dataset.num_ent)]
            elif position == 3:
                return [(r, e1, e2, i, e4, e5) for i in range(1, self.dataset.num_ent)]
            elif position == 4:
                return [(r, e1, e2, e3, i, e5) for i in range(1, self.dataset.num_ent)]
            elif position == 5:
                return [(r, e1, e2, e3, e4, i) for i in range(1, self.dataset.num_ent)]
        if len(fact) == 7:
            r, e1, e2, e3, e4, e5, e6 = fact
            if position == 1:
                return [(r, i, e2, e3, e4, e5, e6) for i in range(1, self.dataset.num_ent)]
            elif position == 2:
                return [(r, e1, i, e3, e4, e5, e6) for i in range(1, self.dataset.num_ent)]
            elif position == 3:
                return [(r, e1, e2, i, e4, e5, e6) for i in range(1, self.dataset.num_ent)]
            elif position == 4:
                return [(r, e1, e2, e3, i, e5, e6) for i in range(1, self.dataset.num_ent)]
            elif position == 5:
                return [(r, e1, e2, e3, e4, i, e6) for i in range(1, self.dataset.num_ent)]
            elif position == 6:
                return [(r, e1, e2, e3, e4, e5, i) for i in range(1, self.dataset.num_ent)]
        if len(fact) == 8:
            r, e1, e2, e3, e4, e5, e6, e7 = fact
            if position == 1:
                return [(r, i, e2, e3, e4, e5, e6, e7) for i in range(1, self.dataset.num_ent)]
            elif position == 2:
                return [(r, e1, i, e3, e4, e5, e6, e7) for i in range(1, self.dataset.num_ent)]
            elif position == 3:
                return [(r, e1, e2, i, e4, e5, e6, e7) for i in range(1, self.dataset.num_ent)]
            elif position == 4:
                return [(r, e1, e2, e3, i, e5, e6, e7) for i in range(1, self.dataset.num_ent)]
            elif position == 5:
                return [(r, e1, e2, e3, e4, i, e6, e7) for i in range(1, self.dataset.num_ent)]
            elif position == 6:
                return [(r, e1, e2, e3, e4, e5, i, e7) for i in range(1, self.dataset.num_ent)]
            elif position == 7:
                return [(r, e1, e2, e3, e4, e5, e6, i) for i in range(1, self.dataset.num_ent)]
        if len(fact) == 9:
            r, e1, e2, e3, e4, e5, e6, e7, e8 = fact
            if position == 1:
                return [(r, i, e2, e3, e4, e5, e6, e7, e8) for i in range(1, self.dataset.num_ent)]
            elif position == 2:
                return [(r, e1, i, e3, e4, e5, e6, e7, e8) for i in range(1, self.dataset.num_ent)]
            elif position == 3:
                return [(r, e1, e2, i, e4, e5, e6, e7, e8) for i in range(1, self.dataset.num_ent)]
            elif position == 4:
                return [(r, e1, e2, e3, i, e5, e6, e7, e8) for i in range(1, self.dataset.num_ent)]
            elif position == 5:
                return [(r, e1, e2, e3, e4, i, e6, e7, e8) for i in range(1, self.dataset.num_ent)]
            elif position == 6:
                return [(r, e1, e2, e3, e4, e5, i, e7, e8) for i in range(1, self.dataset.num_ent)]
            elif position == 7:
                return [(r, e1, e2, e3, e4, e5, e6, i, e8) for i in range(1, self.dataset.num_ent)]
            elif position == 8:
                return [(r, e1, e2, e3, e4, e5, e6, e7, i) for i in range(1, self.dataset.num_ent)]
        if len(fact) == 10:
            r, e1, e2, e3, e4, e5, e6, e7, e8, e9 = fact
            if position == 1:
                return [(r, i, e2, e3, e4, e5, e6, e7, e8, e9) for i in range(1, self.dataset.num_ent)]
            elif position == 2:
                return [(r, e1, i, e3, e4, e5, e6, e7, e8, e9) for i in range(1, self.dataset.num_ent)]
            elif position == 3:
                return [(r, e1, e2, i, e4, e5, e6, e7, e8, e9) for i in range(1, self.dataset.num_ent)]
            elif position == 4:
                return [(r, e1, e2, e3, i, e5, e6, e7, e8, e9) for i in range(1, self.dataset.num_ent)]
            elif position == 5:
                return [(r, e1, e2, e3, e4, i, e6, e7, e8, e9) for i in range(1, self.dataset.num_ent)]
            elif position == 6:
                return [(r, e1, e2, e3, e4, e5, i, e7, e8, e9) for i in range(1, self.dataset.num_ent)]
            elif position == 7:
                return [(r, e1, e2, e3, e4, e5, e6, i, e8, e9) for i in range(1, self.dataset.num_ent)]
            elif position == 8:
                return [(r, e1, e2, e3, e4, e5, e6, e7, i, e9) for i in range(1, self.dataset.num_ent)]
            elif position == 9:
                return [(r, e1, e2, e3, e4, e5, e6, e7, e8, i) for i in range(1, self.dataset.num_ent)]

    def add_fact_and_shred(self, fact, queries, raw_or_fil):
        if raw_or_fil == "raw":
            result = [tuple(fact)] + queries
        elif raw_or_fil == "fil":
            result = [tuple(fact)] + list(set(queries) - self.all_facts_as_set_of_tuples)
        return self.shred_facts(result)


    def test(self, test_by_arity=False):
        """
        Evaluate the given dataset and print results, either by arity or all at once
        """
        settings = ["raw", "fil"]
        normalizer = 0
        self.measure_by_arity = {}
        self.meaddsure = Measure()

        # If the dataset is JF17K and we have test data by arity, then
        # compute test accuracies by arity and show also global result
        if test_by_arity:
        #if (self.valid_or_test == 'test' and self.dataset.data.get('test_2', None) is not None):
            # Iterate over test sets by arity
            for arity in self.ary_list:
                # Reset the normalizer by arity

                print("**** Evaluating arity {} having {} samples".format(arity, len(self.dataset.test[arity])))
                # Evaluate the test data for arity cur_arity
                current_measure, normalizer_by_arity = self.eval_dataset(self.dataset.test[arity])

                # Sum before normalizing current_measure
                normalizer += normalizer_by_arity
                self.measure += current_measure

                # Normalize the values for the current arity and save to dict
                current_measure.normalize(normalizer_by_arity)
                self.measure_by_arity[arity] = current_measure

        else:
            # Evaluate the test data for arity cur_arity
            if self.valid_or_test == "valid":
                current_measure, normalizer = self.eval_dataset(self.dataset.all_valid)
                self.measure = current_measure
            elif self.valid_or_test == "test":
                current_measure, normalizer = self.eval_dataset(self.dataset.all_test)
                self.measure = current_measure

        # If no samples were evaluated, exit with an error
        if normalizer == 0:
            raise Exception("No Samples were evaluated! Check your test or validation data!!")

        # Normalize the global measure
        self.measure.normalize(normalizer)

        # Add the global measure (by ALL arities) to the dict
        self.measure_by_arity["ALL"] = self.measure

        # Print out results
        pr_txt = "Results for ALL ARITIES in {} set".format(self.valid_or_test)
        if test_by_arity:
            for arity in self.measure_by_arity:
                if arity == "ALL":
                    print(pr_txt)
                else:
                    print("Results for arity {}".format(arity))
                print(self.measure_by_arity[arity])
        else:
            print(pr_txt)
            print(self.measure)
        return self.measure, self.measure_by_arity

    def eval_dataset(self, dataset):
        """
        Evaluate the dataset with the given model.
        """
        # Reset normalization parameter
        settings = ["raw", "fil"]
        normalizer = 0
        # Contains the measure values for the given dataset (e.g. test for arity 2)
        current_rank = Measure()
        for i, fact in enumerate(dataset):
            arity = len(fact) - 1
            for j in range(1, arity + 1):
                normalizer += 1
                queries = self.create_queries(fact, j)
                for raw_or_fil in settings:
                    batch = self.add_fact_and_shred(fact, queries, raw_or_fil)


                    sim_scores = self.model.predict(batch).cpu().data.numpy()
                    # Get the rank and update the measures
                    rank = self.get_rank(sim_scores)
                    current_rank.update(rank, raw_or_fil)
                    # self.measure.update(rank, raw_or_fil)

            if i % 1000 == 0:
                print("--- Testing sample {}".format(i))

        return current_rank, normalizer

    def shred_facts(self, tuples):

        tuples = torch.LongTensor(tuples).to(self.device)

        return tuples

    def allFactsAsTuples(self, ary_lst):
        tuples = []
        for arity in ary_lst:
            for train_fact in self.dataset.train[arity]:
                tuples.append(tuple(train_fact))
            for test_fact in self.dataset.test[arity]:
                tuples.append(tuple(test_fact))
            for valid_fact in self.dataset.valid[arity]:
                tuples.append(tuple(valid_fact))
        return tuples

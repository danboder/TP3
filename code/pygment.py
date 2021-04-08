import matplotlib.pyplot as plt


class Pygment:

    def __init__(self, instance_file):
        """
        Represent an instance of the problem
        :param instance_file: file name (string) where the datas are stored
        """

        with open(instance_file) as file:
            lines = file.readlines()

            self.nDays = int(lines[0])  # number of days of the schedule
            self.nProducts = int(lines[1])  # number of products the facility is able to do
            self.shippingSchedule = []  # for each product, the schedule of delivery
            for i in range(self.nProducts):
                sched = [int(k) for k in lines[2 + i].split()]
                self.shippingSchedule.append(sched)
            self.storageCost = int(lines[2 + self.nProducts])  # cost of storage for one unit one day
            self.costMatrix = []  # transition cost matrix of changing from one item to another
            for i in range(self.nProducts):
                cost = [int(k) for k in lines[3 + self.nProducts + i].split()]
                self.costMatrix.append(cost)

    def order(self, i, t):
        """

        :param i: id (int) of the item
        :param t: time stamp (int) of the wanted time
        :return: true if there is an item i ordered for time t
        """
        return self.shippingSchedule[i][t] == 1

    def transition_cost(self, i, j):
        """
        Transition cost from item i to item j
        :param i: id (int) of the 1st item
        :param j: id (int) of the 2nd item
        :return: the cost to configure from item i to item j
        """
        return self.costMatrix[i][j]

    def solution_storage_cost(self, solution):
        """
        Compute the cost of storage done in a solution
        :param solution: the list of produced item (list of id produced)
        :return: the cost (int) of the storage cost of the solution
        """
        sum_date = 0
        cnt = 0
        for i in range(len(solution)):
            if solution[i] >= 0:
                sum_date -= i
                cnt += 1
        for i in self.shippingSchedule:
            for j in range(len(i)):
                if i[j] == 1:
                    sum_date += j
                    cnt -= 1
        cost_s = self.storageCost * sum_date
        return cost_s

    def solution_transition_cost(self, solution):
        """
        Compute the cost of each transition done in a solution
        :param solution: the list of produced item (list of id produced)
        :return: the cost (int) of the transition cost of the solution
        """
        cost_t = 0
        transitions = [i for i in solution if i >= 0]
        for i in range(1, len(transitions)):
            cost_t += self.transition_cost(transitions[i - 1], transitions[i])
        return cost_t

    def solution_total_cost(self, solution):
        """
        The total cost of a solution (transition cost + storage cost)
        :param solution: the list of produced item (list of id produced)
        :return: the cost (int) of the solution
        """
        cost_t = self.solution_transition_cost(solution)
        cost_s = self.solution_storage_cost(solution)
        return cost_t + cost_s

    def display_solution(self, solution, output_file):
        """
        Create a figure displaying the solution
        :param solution: the list of produced item (list of id produced)
        :param output_file: file to store the solution
        """
        fig, ax = plt.subplots(3, figsize=(20, 10))
        x = [i for i in range(self.nDays)]

        prev = [0] * self.nDays
        for i in range(self.nProducts):
            ax[0].bar(x, self.shippingSchedule[i], 0.5, bottom=prev, label='item ' + str(i))
            for j in range(self.nDays):
                prev[j] += self.shippingSchedule[i][j]
        ax[0].set_xlim([-1, self.nDays])
        ax[0].set_title("Shipping schedule")

        for i in range(self.nProducts):
            prod_y = [1 if j == i else 0 for j in solution]
            ax[1].bar(x, prod_y, 0.5, label='item ' + str(i))
        ax[1].set_title("Schedule (total transition cost = {0})".format(self.solution_transition_cost(solution)))
        ax[1].set_xlim([-1, self.nDays])

        storage_y = [0 for i in range(self.nDays)]
        for i in range(self.nDays):
            if solution[i] >= 0:
                storage_y[i] += 1
            for j in self.shippingSchedule:
                if j[i] == 1:
                    storage_y[i] -= 1
            if i != 0:
                storage_y[i] += storage_y[i - 1]
        ax[2].set_title("Items in storage (total storage cost = {0})".format(self.solution_storage_cost(solution)))
        ax[2].plot(x, storage_y)
        ax[2].set_xlim([-1, self.nDays])

        plt.savefig(output_file)

    def print_solution(self, solution, output_file):
        with open(output_file, "w") as file:
            file.write(str(self.solution_total_cost(solution)) + "\n")
            for i in solution:
                file.write(str(i) + " ")

    def verify_solution(self, solution, display=False):
        for i in range(self.nProducts):
            s = [j for j in range(self.nDays) if self.order(i, j)]
            p = [j for j in range(self.nDays) if solution[j] == i]
            if len(s) != len(p):
                if display: print("your solution produces {0} items of id {1}, but {2} were asked".format(len(p), i, len(s)))
                return False
            for k in range(len(s)):
                if s[k] < p[k]:
                    if display: print("your solution produces some item {0} too late for delivery".format(i))
                    return False
        return True

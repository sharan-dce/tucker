import random
import os
import csv


def take(l):
  # Remove a randomly selected element from the list,
  # and return a tuple (removed_element, list_of_others)
  idx = random.randint(0, len(l)-1)
  elem = l[idx]
  l.remove(elem)
  return elem, l

  
def gen_inverses(n_e=10000, p_test=0.3, rel_1=0, rel_2=1):
  # Generate a synthetic dataset with an inverse pair
  # of relations
  train = []
  test = []
  for i in range(0, n_e, 2):
    facts = [(i, rel_1, i+1), (i+1, rel_2, i)]
    if random.random() < p_test:
      test_fact, train_facts = take(facts)
      test.append(test_fact)
      train += train_facts
    else:
      train += facts
  return train, test


def gen_symmetries(n_e=10000, p_test=0.3):
  # Generate a synthetic dataset with a symmetric relation
  return gen_inverses(n_e=n_e, p_test=p_test, rel_1=0, rel_2=0)


def gen_compositions(n_e=9999, p_test=0.3):
  # Generate a synthetic dataset with composing relations
  train = []
  test = []
  for i in range(0, n_e, 3):
    facts = [(i, 0, i+1), (i+1, 1, i+2), (i, 2, i+2)]
    if random.random() < p_test:
      test_fact, train_facts = take(facts)
      test.append(test_fact)
      train += train_facts
    else:
      train += facts
  return train, test


def gen_hierarchies(n_e=10008, p_test=0.5):
  # Generate a synthetic dataset with a non-trivial
  # hierarchy r1 => r2
  train = []
  test = []
  for i in range(0, n_e//2, 4):
    facts = [(i, 0, i+1), (i+1, 1, i), (i, 2, i+1), (i+2, 2, i+3)]
    train += facts
  for i in range(n_e//2, n_e, 3):
    facts = [(i, 0, i+1), (i+1, 1, i), (i, 2, i+2), (i, 2, i+1)]
    if random.random() > p_test:
      test_fact = facts[0]
      train_facts = facts[1:]
      test.append(test_fact)
      train += train_facts
    else:
      train += facts
  return train, test

  
def to_csv(facts, file):
  writer = csv.writer(file, delimiter='\t')
  for fact in facts:
    writer.writerow(fact)

	
def to_csvs(train_facts, test_facts, name):
  folder = os.path.join('.', 'data', name)
  os.makedirs(folder)
  train_fn = os.path.join(folder, 'train.txt')
  test_fn = os.path.join(folder, 'test.txt')
  valid_fn = os.path.join(folder, 'valid.txt')
  with open(train_fn, 'w', newline='') as train_csvfile:
    to_csv(train_facts, train_csvfile)
  with open(test_fn, 'w', newline='') as test_csvfile:
    to_csv(test_facts, test_csvfile)
  with open(valid_fn, 'w', newline='') as valid_csvfile:
    to_csv(test_facts, valid_csvfile)
	
if __name__=='__main__':
  train, test = gen_inverses()
  to_csvs(train, test, 'inversion')
  train, test = gen_symmetries()
  to_csvs(train, test, 'symmetry')
  train, test = gen_compositions()
  to_csvs(train, test, 'composition')
  train, test = gen_hierarchies()
  to_csvs(train, test, 'hierarchy')
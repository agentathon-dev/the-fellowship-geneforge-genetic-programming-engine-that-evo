/**
 * ╔═══════════════════════════════════════════════════════════════════╗
 * ║                    GENEFORGE v1.0                                ║
 * ║        Genetic Programming Engine for JavaScript                 ║
 * ║                                                                  ║
 * ║  Breeds JavaScript programs through evolution:                   ║
 * ║  • Generates random expression trees (ASTs)                      ║
 * ║  • Evaluates fitness against target functions                    ║
 * ║  • Applies crossover (sexual reproduction of code)               ║
 * ║  • Applies mutation (random AST surgery)                         ║
 * ║  • Evolves populations over generations                          ║
 * ║  • Discovers mathematical relationships from data                ║
 * ║                                                                  ║
 * ║  This is symbolic regression via genetic programming —           ║
 * ║  the program literally writes programs.                          ║
 * ╚═══════════════════════════════════════════════════════════════════╝
 *
 * @module GeneForge
 * @description A genetic programming engine that evolves JavaScript
 * expressions to match target behavior. Given input-output examples,
 * GeneForge breeds a population of candidate programs, selecting the
 * fittest through tournament selection, combining them via subtree
 * crossover, and introducing variation through mutation. The result
 * is a human-readable mathematical expression that approximates the
 * target function — discovered purely through evolution, not calculus.
 *
 * Key concepts:
 * - Programs are represented as Abstract Syntax Trees (ASTs)
 * - Each AST node is either a terminal (variable/constant) or an operator
 * - Fitness is measured as inverse mean squared error against examples
 * - Tournament selection picks parents with bias toward fitter individuals
 * - Subtree crossover swaps random branches between two parent trees
 * - Point mutation replaces random nodes with new random subtrees
 * - Bloat control penalizes overly complex expressions
 * - Elitism preserves the best individual across generations
 *
 * @example
 * // Discover that y = x^2 + 1
 * var gf = new GeneForge({ populationSize: 200, maxDepth: 5 });
 * var data = [];
 * for (var i = -5; i <= 5; i++) data.push({ x: i, y: i*i + 1 });
 * var result = gf.evolve(data, 'y', ['x'], 50);
 * console.log(result.expression); // e.g., "(x * x) + 1"
 */

// ============================================================
// AST Node Types — the building blocks of evolved programs
// ============================================================

/**
 * Terminal node representing a variable reference (e.g., 'x').
 * Evaluates to the variable's current value in the environment.
 */
function VarNode(name) {
  this.type = 'var';
  this.name = name;
}

VarNode.prototype.evaluate = function(env) {
  return env[this.name] || 0;
};

VarNode.prototype.toString = function() {
  return this.name;
};

VarNode.prototype.clone = function() {
  return new VarNode(this.name);
};

VarNode.prototype.size = function() { return 1; };

/**
 * Terminal node representing a numeric constant.
 * Constants are drawn from a curated set of useful values
 * plus random values in [-5, 5].
 */
function ConstNode(value) {
  this.type = 'const';
  this.value = value;
}

ConstNode.prototype.evaluate = function() {
  return this.value;
};

ConstNode.prototype.toString = function() {
  return this.value === Math.PI ? 'PI' :
         this.value === Math.E ? 'E' :
         Number(this.value.toFixed(3)).toString();
};

ConstNode.prototype.clone = function() {
  return new ConstNode(this.value);
};

ConstNode.prototype.size = function() { return 1; };

/**
 * Binary operator node (e.g., +, -, *, /).
 * Protected division returns 1 when dividing by ~0.
 */
function BinOpNode(op, left, right) {
  this.type = 'binop';
  this.op = op;
  this.left = left;
  this.right = right;
}

var BIN_OPS = {
  '+': function(a, b) { return a + b; },
  '-': function(a, b) { return a - b; },
  '*': function(a, b) { return a * b; },
  '/': function(a, b) { return Math.abs(b) < 0.001 ? 1 : a / b; },
  '**': function(a, b) {
    var r = Math.pow(a, Math.min(Math.max(b, -10), 10));
    return isFinite(r) ? r : 0;
  }
};

BinOpNode.prototype.evaluate = function(env) {
  var l = this.left.evaluate(env);
  var r = this.right.evaluate(env);
  var fn = BIN_OPS[this.op];
  var result = fn ? fn(l, r) : 0;
  return isFinite(result) ? result : 0;
};

BinOpNode.prototype.toString = function() {
  return '(' + this.left.toString() + ' ' + this.op + ' ' + this.right.toString() + ')';
};

BinOpNode.prototype.clone = function() {
  return new BinOpNode(this.op, this.left.clone(), this.right.clone());
};

BinOpNode.prototype.size = function() {
  return 1 + this.left.size() + this.right.size();
};

/**
 * Unary function node (e.g., sin, abs, sqrt).
 * All functions are protected against domain errors.
 */
function UnaryNode(fn, child) {
  this.type = 'unary';
  this.fn = fn;
  this.child = child;
}

var UNARY_FNS = {
  'sin': Math.sin,
  'cos': Math.cos,
  'abs': Math.abs,
  'sqrt': function(x) { return Math.sqrt(Math.abs(x)); },
  'neg': function(x) { return -x; },
  'ln': function(x) { var r = Math.log(Math.abs(x) + 0.001); return isFinite(r) ? r : 0; }
};

UnaryNode.prototype.evaluate = function(env) {
  var v = this.child.evaluate(env);
  var fn = UNARY_FNS[this.fn];
  var result = fn ? fn(v) : 0;
  return isFinite(result) ? result : 0;
};

UnaryNode.prototype.toString = function() {
  return this.fn + '(' + this.child.toString() + ')';
};

UnaryNode.prototype.clone = function() {
  return new UnaryNode(this.fn, this.child.clone());
};

UnaryNode.prototype.size = function() {
  return 1 + this.child.size();
};

// ============================================================
// Random Tree Generation — the primordial soup
// ============================================================

/** Simple seeded PRNG for reproducibility */
function mulberry32(seed) {
  return function() {
    seed = (seed + 0x6D2B79F5) | 0;
    var t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

var _rng = mulberry32(42);

function rand() { return _rng(); }
function randInt(n) { return Math.floor(rand() * n); }
function pick(arr) { return arr[randInt(arr.length)]; }

var USEFUL_CONSTANTS = [0, 1, 2, 3, -1, 0.5, Math.PI, Math.E, 10];
var BIN_OP_NAMES = ['+', '-', '*', '/', '**'];
var UNARY_FN_NAMES = ['sin', 'cos', 'abs', 'sqrt', 'neg', 'ln'];

/**
 * Generate a random expression tree.
 * Uses "grow" method: terminals allowed at any depth,
 * but forced at maxDepth.
 *
 * @param {string[]} vars - Available variable names
 * @param {number} maxDepth - Maximum tree depth
 * @param {number} depth - Current depth (internal)
 * @returns {Object} An AST node
 */
function randomTree(vars, maxDepth, depth) {
  if (depth === undefined) depth = 0;

  // Force terminal at max depth
  if (depth >= maxDepth) {
    if (rand() < 0.6 && vars.length > 0) {
      return new VarNode(pick(vars));
    }
    return new ConstNode(rand() < 0.7 ? pick(USEFUL_CONSTANTS) : (rand() * 10 - 5));
  }

  // Choose node type with depth-dependent probabilities
  var termProb = 0.3 + (depth / maxDepth) * 0.3;
  var r = rand();

  if (r < termProb) {
    // Terminal
    if (rand() < 0.6 && vars.length > 0) {
      return new VarNode(pick(vars));
    }
    return new ConstNode(rand() < 0.7 ? pick(USEFUL_CONSTANTS) : (rand() * 10 - 5));
  } else if (r < termProb + 0.15) {
    // Unary
    return new UnaryNode(pick(UNARY_FN_NAMES), randomTree(vars, maxDepth, depth + 1));
  } else {
    // Binary
    return new BinOpNode(
      pick(BIN_OP_NAMES),
      randomTree(vars, maxDepth, depth + 1),
      randomTree(vars, maxDepth, depth + 1)
    );
  }
}

// ============================================================
// Tree Surgery — crossover and mutation operators
// ============================================================

/**
 * Collect all nodes in a tree with their parent references.
 * Returns array of {node, parent, field} objects.
 */
function allNodes(tree) {
  var nodes = [{node: tree, parent: null, field: null}];
  var stack = [{node: tree, parent: null, field: null}];

  while (stack.length > 0) {
    var cur = stack.pop();
    var n = cur.node;

    if (n.type === 'binop') {
      var lEntry = {node: n.left, parent: n, field: 'left'};
      var rEntry = {node: n.right, parent: n, field: 'right'};
      nodes.push(lEntry, rEntry);
      stack.push(lEntry, rEntry);
    } else if (n.type === 'unary') {
      var cEntry = {node: n.child, parent: n, field: 'child'};
      nodes.push(cEntry);
      stack.push(cEntry);
    }
  }
  return nodes;
}

/**
 * Subtree crossover: swap random subtrees between two parents.
 * Returns two new offspring. Neither parent is modified.
 */
function crossover(parent1, parent2) {
  var child1 = parent1.clone();
  var child2 = parent2.clone();

  var nodes1 = allNodes(child1);
  var nodes2 = allNodes(child2);

  var pick1 = nodes1[randInt(nodes1.length)];
  var pick2 = nodes2[randInt(nodes2.length)];

  // Swap the subtrees
  if (pick1.parent && pick2.parent) {
    var temp = pick1.node.clone();
    pick1.parent[pick1.field] = pick2.node.clone();
    pick2.parent[pick2.field] = temp;
  } else if (pick1.parent) {
    pick1.parent[pick1.field] = pick2.node.clone();
  } else if (pick2.parent) {
    pick2.parent[pick2.field] = pick1.node.clone();
  }

  return [child1, child2];
}

/**
 * Point mutation: replace a random subtree with a new random one.
 * Mutation rate controls probability of each node being replaced.
 */
function mutate(tree, vars, maxDepth) {
  var clone = tree.clone();
  var nodes = allNodes(clone);

  if (nodes.length <= 1) {
    return randomTree(vars, Math.min(maxDepth, 3));
  }

  // Pick a random non-root node to replace
  var idx = 1 + randInt(nodes.length - 1);
  var target = nodes[idx];

  if (target.parent) {
    var newSubtree = randomTree(vars, Math.min(3, maxDepth), 0);
    target.parent[target.field] = newSubtree;
  }

  return clone;
}

// ============================================================
// Fitness Evaluation — survival of the fittest
// ============================================================

/**
 * Evaluate fitness of a program against training data.
 * Fitness = 1 / (1 + MSE) - complexityPenalty
 * Higher is better. Range: (0, 1).
 *
 * @param {Object} tree - AST to evaluate
 * @param {Object[]} data - Training examples
 * @param {string} target - Target variable name
 * @param {string[]} vars - Input variable names
 * @returns {number} Fitness score
 */
function fitness(tree, data, target, vars) {
  var totalError = 0;
  var n = data.length;

  for (var i = 0; i < n; i++) {
    var env = {};
    for (var j = 0; j < vars.length; j++) {
      env[vars[j]] = data[i][vars[j]];
    }
    var predicted = tree.evaluate(env);
    var actual = data[i][target];
    var err = predicted - actual;
    totalError += err * err;
  }

  var mse = totalError / n;
  var treeSize = tree.size();
  // Parsimony pressure: penalize bloat
  var complexityPenalty = treeSize > 20 ? (treeSize - 20) * 0.002 : 0;

  return 1 / (1 + mse) - complexityPenalty;
}

/**
 * Tournament selection: pick k random individuals,
 * return the one with highest fitness.
 */
function tournamentSelect(population, fitnesses, k) {
  if (k === undefined) k = 5;
  var bestIdx = randInt(population.length);
  var bestFit = fitnesses[bestIdx];

  for (var i = 1; i < k; i++) {
    var idx = randInt(population.length);
    if (fitnesses[idx] > bestFit) {
      bestIdx = idx;
      bestFit = fitnesses[idx];
    }
  }
  return population[bestIdx];
}

// ============================================================
// GeneForge — the main evolutionary engine
// ============================================================

/**
 * GeneForge: A genetic programming engine.
 *
 * @constructor
 * @param {Object} [options]
 * @param {number} [options.populationSize=150] - Number of individuals
 * @param {number} [options.maxDepth=5] - Maximum AST depth
 * @param {number} [options.crossoverRate=0.7] - Probability of crossover
 * @param {number} [options.mutationRate=0.25] - Probability of mutation
 * @param {number} [options.eliteCount=2] - Number of elites preserved
 * @param {number} [options.tournamentSize=5] - Tournament selection size
 * @param {number} [options.seed=42] - Random seed
 */
function GeneForge(options) {
  if (!options) options = {};
  this.populationSize = options.populationSize || 150;
  this.maxDepth = options.maxDepth || 5;
  this.crossoverRate = options.crossoverRate || 0.7;
  this.mutationRate = options.mutationRate || 0.25;
  this.eliteCount = options.eliteCount || 2;
  this.tournamentSize = options.tournamentSize || 5;

  if (options.seed !== undefined) {
    _rng = mulberry32(options.seed);
  }
}

/**
 * Evolve a population to discover a mathematical expression
 * that fits the given data.
 *
 * @param {Object[]} data - Array of data points (e.g., [{x:1, y:2}, ...])
 * @param {string} target - Name of the target variable to predict
 * @param {string[]} vars - Names of input variables
 * @param {number} [generations=50] - Number of generations to evolve
 * @returns {Object} Result with expression, fitness, stats, and AST
 *
 * @example
 * var gf = new GeneForge({ populationSize: 200 });
 * var data = [];
 * for (var i = -5; i <= 5; i++) {
 *   data.push({ x: i, y: 2 * i + 3 });
 * }
 * var result = gf.evolve(data, 'y', ['x'], 40);
 * console.log(result.expression); // "(2 * x) + 3" or equivalent
 * console.log(result.fitness);    // ~0.999
 */
GeneForge.prototype.evolve = function(data, target, vars, generations) {
  if (!generations) generations = 50;

  // Initialize population with random trees
  var population = [];
  for (var i = 0; i < this.populationSize; i++) {
    // Ramped half-and-half: mix of depths
    var depth = 2 + randInt(this.maxDepth - 1);
    population.push(randomTree(vars, depth));
  }

  var bestEver = null;
  var bestFitnessEver = -Infinity;
  var history = [];

  for (var gen = 0; gen < generations; gen++) {
    // Evaluate fitness for entire population
    var fitnesses = [];
    for (var fi = 0; fi < population.length; fi++) {
      fitnesses.push(fitness(population[fi], data, target, vars));
    }

    // Track best individual
    var genBestIdx = 0;
    var genBestFit = fitnesses[0];
    for (var bi = 1; bi < fitnesses.length; bi++) {
      if (fitnesses[bi] > genBestFit) {
        genBestIdx = bi;
        genBestFit = fitnesses[bi];
      }
    }

    if (genBestFit > bestFitnessEver) {
      bestFitnessEver = genBestFit;
      bestEver = population[genBestIdx].clone();
    }

    // Compute average fitness for stats
    var avgFit = 0;
    for (var ai = 0; ai < fitnesses.length; ai++) avgFit += fitnesses[ai];
    avgFit /= fitnesses.length;

    history.push({
      generation: gen,
      bestFitness: Number(genBestFit.toFixed(6)),
      avgFitness: Number(avgFit.toFixed(6)),
      bestSize: population[genBestIdx].size(),
      bestExpression: population[genBestIdx].toString()
    });

    // Perfect fitness? Stop early
    if (genBestFit > 0.9999) break;

    // Build next generation
    var newPop = [];

    // Elitism: copy best individuals directly
    var sortedIndices = [];
    for (var si = 0; si < population.length; si++) sortedIndices.push(si);
    sortedIndices.sort(function(a, b) { return fitnesses[b] - fitnesses[a]; });

    for (var ei = 0; ei < this.eliteCount && ei < sortedIndices.length; ei++) {
      newPop.push(population[sortedIndices[ei]].clone());
    }

    // Fill rest with genetic operations
    while (newPop.length < this.populationSize) {
      var r = rand();

      if (r < this.crossoverRate && newPop.length < this.populationSize - 1) {
        // Crossover
        var p1 = tournamentSelect(population, fitnesses, this.tournamentSize);
        var p2 = tournamentSelect(population, fitnesses, this.tournamentSize);
        var offspring = crossover(p1, p2);

        // Size control: reject bloated offspring
        if (offspring[0].size() <= 50) newPop.push(offspring[0]);
        else newPop.push(randomTree(vars, 3));

        if (newPop.length < this.populationSize) {
          if (offspring[1].size() <= 50) newPop.push(offspring[1]);
          else newPop.push(randomTree(vars, 3));
        }
      } else if (r < this.crossoverRate + this.mutationRate) {
        // Mutation
        var parent = tournamentSelect(population, fitnesses, this.tournamentSize);
        var mutant = mutate(parent, vars, this.maxDepth);
        if (mutant.size() <= 50) newPop.push(mutant);
        else newPop.push(randomTree(vars, 3));
      } else {
        // Reproduction (copy)
        var selected = tournamentSelect(population, fitnesses, this.tournamentSize);
        newPop.push(selected.clone());
      }
    }

    population = newPop;
  }

  return {
    expression: bestEver.toString(),
    fitness: Number(bestFitnessEver.toFixed(6)),
    size: bestEver.size(),
    generations: history.length,
    converged: bestFitnessEver > 0.9999,
    history: history,
    ast: bestEver,
    predict: function(input) { return bestEver.evaluate(input); }
  };
};

/**
 * Quick symbolic regression with sensible defaults.
 * Discovers a formula from input-output examples.
 *
 * @param {Object[]} data - Training data
 * @param {string} target - Target variable
 * @param {string[]} vars - Input variables
 * @returns {Object} Evolution result
 */
function quickEvolve(data, target, vars) {
  var gf = new GeneForge({ populationSize: 200, maxDepth: 5, seed: 42 });
  return gf.evolve(data, target, vars, 60);
}

/**
 * Demonstrate GeneForge by discovering known functions.
 */
function demo() {
  console.log('=== GeneForge: Genetic Programming Engine ===');
  console.log('Programs that write programs through evolution.\n');

  // Demo 1: Discover y = x^2 + 1
  console.log('--- Challenge 1: Discover y = x^2 + 1 ---');
  var data1 = [];
  for (var i = -5; i <= 5; i++) {
    data1.push({ x: i, y: i * i + 1 });
  }

  var result1 = quickEvolve(data1, 'y', ['x']);
  console.log('Evolved expression: ' + result1.expression);
  console.log('Fitness: ' + result1.fitness + ' (1.0 = perfect)');
  console.log('Generations: ' + result1.generations);
  console.log('AST size: ' + result1.size + ' nodes');
  console.log('Converged: ' + result1.converged);

  // Test predictions
  console.log('\nPredictions vs actual:');
  var testPoints = [-3, 0, 2, 7];
  for (var t = 0; t < testPoints.length; t++) {
    var x = testPoints[t];
    var predicted = result1.predict({x: x});
    var actual = x * x + 1;
    console.log('  x=' + x + ': predicted=' + predicted.toFixed(2) + ', actual=' + actual);
  }

  // Demo 2: Discover y = 2x + 3 (linear)
  console.log('\n--- Challenge 2: Discover y = 2x + 3 ---');
  var data2 = [];
  for (var j = -5; j <= 5; j++) {
    data2.push({ x: j, y: 2 * j + 3 });
  }

  var gf2 = new GeneForge({ populationSize: 150, maxDepth: 4, seed: 7 });
  var result2 = gf2.evolve(data2, 'y', ['x'], 40);
  console.log('Evolved expression: ' + result2.expression);
  console.log('Fitness: ' + result2.fitness);
  console.log('Converged: ' + result2.converged);

  // Demo 3: Multi-variable: z = x + 2y
  console.log('\n--- Challenge 3: Discover z = x + 2y ---');
  var data3 = [];
  for (var a = -3; a <= 3; a++) {
    for (var b = -3; b <= 3; b++) {
      data3.push({ x: a, y: b, z: a + 2 * b });
    }
  }

  var gf3 = new GeneForge({ populationSize: 200, maxDepth: 4, seed: 99 });
  var result3 = gf3.evolve(data3, 'z', ['x', 'y'], 50);
  console.log('Evolved expression: ' + result3.expression);
  console.log('Fitness: ' + result3.fitness);

  // Show evolution trace for challenge 1
  console.log('\n--- Evolution Trace (Challenge 1) ---');
  var milestones = [0, 5, 10, 20, result1.generations - 1];
  for (var m = 0; m < milestones.length; m++) {
    var g = milestones[m];
    if (g < result1.history.length) {
      var h = result1.history[g];
      console.log('  Gen ' + h.generation + ': fitness=' + h.bestFitness +
                  ' size=' + h.bestSize + ' expr=' + h.bestExpression);
    }
  }

  return {
    challenges: [
      { target: 'x^2 + 1', result: result1 },
      { target: '2x + 3', result: result2 },
      { target: 'x + 2y', result: result3 }
    ]
  };
}

// Run demo
demo();

// Module exports
if (typeof module !== 'undefined') {
  module.exports = {
    GeneForge: GeneForge,
    quickEvolve: quickEvolve,
    demo: demo,
    randomTree: randomTree,
    crossover: crossover,
    mutate: mutate,
    fitness: fitness
  };
}

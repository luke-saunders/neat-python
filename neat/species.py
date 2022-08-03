"""Divides the population into species based on genomic distances."""
from itertools import count

from neat.config import ConfigParameter, DefaultClassConfig
from neat.math_util import mean, stdev


class Species(object):
    def __init__(self, key, generation):
        self.key = key
        self.created = generation
        self.last_improved = generation
        self.representative = None
        self.members = {}
        self.fitness = None
        self.adjusted_fitness = None
        self.fitness_history = []

    def update(self, representative, members):
        self.representative = representative
        self.members = members

    def get_fitnesses(self):
        return [m.fitness for m in self.members.values()]


class GenomeDistanceCache(object):
    def __init__(self, config):
        self.distances = {}
        self.config = config
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0, genome1):
        g0 = genome0.key
        g1 = genome1.key
        d = self.distances.get((g0, g1))
        if d is None:
            # Distance is not already computed.
            d = genome0.distance(genome1, self.config)
            self.distances[g0, g1] = d
            self.distances[g1, g0] = d
            self.misses += 1
        else:
            self.hits += 1

        return d


class DefaultSpeciesSet(DefaultClassConfig):
    """Encapsulates the default speciation scheme."""

    def __init__(self, config, reporters):
        # pylint: disable=super-init-not-called
        self.species_set_config = config
        self.reporters = reporters
        self.indexer = count(1)
        self.species = {}
        self.genome_to_species = {}
        self.dynamic_compatibility_threshold = (
            self.species_set_config.compatibility_threshold_init
        )

    @classmethod
    def parse_config(cls, param_dict):
        return DefaultClassConfig(
            param_dict,
            [
                ConfigParameter("number_of_species_min", int),
                ConfigParameter("number_of_species_max", int),
                ConfigParameter("compatibility_threshold_init", float),
                ConfigParameter("compatibility_threshold_min", float),
                ConfigParameter("compatibility_threshold_max", float),
                ConfigParameter("compatibility_modifier", float),
                ConfigParameter("review_frequency_inc", int),
                ConfigParameter("review_frequency_dec", int),
            ],
        )

    def speciate(self, config, population, generation):
        """
        Place genomes into species by genetic similarity.

        Note that this method assumes the current representatives of the species are from the old
        generation, and that after speciation has been performed, the old representatives should be
        dropped and replaced with representatives from the new generation.  If you violate this
        assumption, you should make sure other necessary parts of the code are updated to reflect
        the new behavior.
        """
        assert isinstance(population, dict)

        compatibility_adj = 0
        number_of_species = len(self.species)

        # [min, max]
        target_number_of_species = [0, 0]
        # target number of species - lower limit
        target_number_of_species[0] = self.species_set_config.number_of_species_min
        # target number of species - upper limit
        target_number_of_species[1] = self.species_set_config.number_of_species_max
        # compatibility_threshold - lower limit
        compatibility_threshold_min = (
            self.species_set_config.compatibility_threshold_min
        )
        # compatibility_threshold - upper limit
        compatibility_threshold_max = (
            self.species_set_config.compatibility_threshold_max
        )
        # modifier weight compatibility change weight
        compatibility_modifier = self.species_set_config.compatibility_modifier
        # frequency of value review, for number_of_species too low
        review_frequency_inc = self.species_set_config.review_frequency_inc
        # frequency of value review, for number_of_species too high
        review_frequency_dec = self.species_set_config.review_frequency_dec

        # Fine tune the compatibility threshold
        if (number_of_species < target_number_of_species["min"]) and (
            (generation + 1) % review_frequency_inc == 0
        ):
            # Number of species is too low
            num_species_gap = number_of_species - mean(target_number_of_species)
            compatibility_adj = compatibility_modifier * num_species_gap
            self.dynamic_compatibility_threshold += compatibility_adj
            # Clamp min value
            self.dynamic_compatibility_threshold = max(
                self.dynamic_compatibility_threshold, compatibility_threshold_min
            )

        elif (number_of_species > target_number_of_species["max"]) and (
            (generation + 1) % review_frequency_dec == 0
        ):
            # Number of species is too high
            num_species_gap = number_of_species - mean(target_number_of_species)
            compatibility_adj = compatibility_modifier * num_species_gap
            self.dynamic_compatibility_threshold += compatibility_adj
            # Clamp max value
            self.dynamic_compatibility_threshold = min(
                self.dynamic_compatibility_threshold, compatibility_threshold_max
            )

        if compatibility_adj >= 0:
            comp_adj = "+" + str(round(compatibility_adj, 4))
        else:
            comp_adj = str(round(compatibility_adj, 4))
        self.reporters.info(
            "Compatibility Threshold {0:.3f} ({1})".format(
                self.dynamic_compatibility_threshold, comp_adj
            )
        )

        # Find the best representatives for each existing species.
        unspeciated = set(population)
        distances = GenomeDistanceCache(config.genome_config)
        new_representatives = {}
        new_members = {}
        for sid, s in self.species.items():
            candidates = []
            for gid in unspeciated:
                g = population[gid]
                d = distances(s.representative, g)
                candidates.append((d, g))

            # The new representative is the genome closest to the current representative.
            ignored_rdist, new_rep = min(candidates, key=lambda x: x[0])
            new_rid = new_rep.key
            new_representatives[sid] = new_rid
            new_members[sid] = [new_rid]
            unspeciated.remove(new_rid)

        # Partition population into species based on genetic similarity.
        while unspeciated:
            gid = unspeciated.pop()
            g = population[gid]

            # Find the species with the most similar representative.
            candidates = []
            for sid, rid in new_representatives.items():
                rep = population[rid]
                d = distances(rep, g)
                if d < self.dynamic_compatibility_threshold:
                    candidates.append((d, sid))

            if candidates:
                ignored_sdist, sid = min(candidates, key=lambda x: x[0])
                new_members[sid].append(gid)
            else:
                # No species is similar enough, create a new species, using
                # this genome as its representative.
                sid = next(self.indexer)
                new_representatives[sid] = gid
                new_members[sid] = [gid]

        # Update species collection based on new speciation.
        self.genome_to_species = {}
        for sid, rid in new_representatives.items():
            s = self.species.get(sid)
            if s is None:
                s = Species(sid, generation)
                self.species[sid] = s

            members = new_members[sid]
            for gid in members:
                self.genome_to_species[gid] = sid

            member_dict = dict((gid, population[gid]) for gid in members)
            s.update(population[rid], member_dict)

        gdmean = mean(distances.distances.values())
        gdstdev = stdev(distances.distances.values())
        self.reporters.info(
            "Mean genetic distance {0:.3f}, standard deviation {1:.3f}".format(
                gdmean, gdstdev
            )
        )

    def get_species_id(self, individual_id):
        return self.genome_to_species[individual_id]

    def get_species(self, individual_id):
        sid = self.genome_to_species[individual_id]
        return self.species[sid]

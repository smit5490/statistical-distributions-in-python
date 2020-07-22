import statdist


#### Tests for Gaussian Distribution ####

class TestGaussian:
    def test_instantiation(self):
        dist = statdist.Gaussian(mu=25, sigma=2)
        assert dist.mean == 25, "mean attribute incorrect"
        assert dist.stdev == 2, "st. dev. incorrect."
        assert str(dist) == "mean 25, standard deviation 2", "Object call print incorrect"

    def test_data_functions(self):
        dist = statdist.Gaussian(mu=25, sigma=2)
        dist.read_data_file("numbers.txt")
        assert dist.data == [1, 3, 99, 100, 120, 32, 330, 23, 76, 44, 31], "read_data_file method incorrect"
        assert dist.calculate_mean() == sum(dist.data) / float(len(dist.data)), "calculate_mean method incorrect"
        assert round(dist.calculate_stdev(sample=True), 2) == 92.87, "calculate_stdev method with sample incorrect"
        assert round(dist.calculate_stdev(sample=False), 2) == 88.55, "calculate_stdev w/o sample incorrect"

    def test_pdf(self):
        dist = statdist.Gaussian(mu=25, sigma=2)
        assert round(dist.pdf(25), 5) == 0.19947, 'pdf function does not give expected result'

        # test pdf with numbers file
        dist.read_data_file("numbers.txt")
        dist.calculate_mean()
        dist.calculate_stdev()
        assert round(dist.pdf(75),
                     5) == 0.00429, 'pdf function after calculating mean and stdev does not give expected result'

    def test_cum_pdf(self):
        dist = statdist.Gaussian(mu=25, sigma=2)
        assert round(dist.cum_pdf(27), 2) == .84, "cum_pdf method incorrect"
        assert round(dist.cum_pdf(27, lower=False), 2) == .16, "cum_pdf method incorrect"

    def test_z_score(self):
        dist = statdist.Gaussian(mu=25, sigma=2)
        assert dist.z_score(27) == 1.0, "z_score method incorrect"

    def test_sum(self):
        dist_one = statdist.Gaussian(25, 3)
        dist_two = statdist.Gaussian(30, 4)
        dist_sum = dist_one + dist_two

        assert dist_sum.mean == 55, "mean of added distributions incorrect"
        assert dist_sum.stdev == 5, "st. dev of added distributions incorrect"


#### Tests for Gaussian Distribution ###
class TestBinomial:

    def test_initialization(self):
        binomial_dist = statdist.Binomial(0.4, 20)
        assert binomial_dist.p == 0.4, 'p value incorrect'
        assert binomial_dist.n == 20, 'n value incorrect'

    def test_read_data(self):
        binomial_dist = statdist.Binomial()
        binomial_dist.read_data_file("numbers_binomial.txt")
        assert binomial_dist.data == [0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0], 'data not read in correctly'

    def test_calculate_mean(self):
        binomial_dist = statdist.Binomial(0.4, 20)
        mean = binomial_dist.calculate_mean()
        assert mean == 8, "calculate_mean method incorrect"

    def test_calculate_stdev(self):
        binomial_dist = statdist.Binomial(0.4, 20)
        stdev = binomial_dist.calculate_stdev()
        assert round(stdev, 2) == 2.19, "calculate stdev method incorrect"

    def test_pdf(self):
        binomial_dist = statdist.Binomial(0.4, 20)
        binomial_dist.read_data_file("numbers_binomial.txt")
        assert round(binomial_dist.pdf(5), 5) == 0.07465, "pdf method incorrect"
        assert round(binomial_dist.pdf(3), 5) == 0.01235, "pdf method incorrect"

        binomial_dist.replace_stats_with_data()
        assert round(binomial_dist.pdf(5), 5) == 0.05439, "pdf method after replace_stats_with data method is incorrect"
        assert round(binomial_dist.pdf(3), 5) == 0.00472, "pdf method after replace_stats_with data method is incorrect"

    def test_replace_stats_with_data(self):
        binomial_dist = statdist.Binomial()
        binomial_dist.read_data_file("numbers_binomial.txt")
        n, p = binomial_dist.replace_stats_with_data()
        assert round(p, 3) == .615, "replace_stats_with_data method incorrect"
        assert n == 13, "replace_stats_with_data method incorrect"

    def test_add(self):
        binomial_one = statdist.Binomial(.4, 20)
        binomial_two = statdist.Binomial(.4, 60)
        binomial_sum = binomial_one + binomial_two

        assert binomial_sum.p == .4, "adding distributions capability incorrect"
        assert binomial_sum.n == 80, "adding distributions capability incorrect"

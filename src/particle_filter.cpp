/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *      Updated by: Danilo Canivel
 *      Update date: Nov 13, 2017
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
    //   x, y, theta and their uncertainties from GPS) and all weights to 1.
    // Add random Gaussian noise to each particle.
    // NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    // define noise
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_theta(theta, std[2]);

    for (int i = 0; i < num_particles; ++i)
    {
        Particle p;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;
        particles.push_back(p);

        weights.push_back(1.0);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // TODO: Add measurements to each particle and add random Gaussian noise.
    // NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
    //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
    //  http://www.cplusplus.com/reference/random/default_random_engine/

    for (int i = 0; i < num_particles; i++)
    {
        double x, y, theta;

        if (fabs(yaw_rate) < 0.0001){
            x = particles[i].x + velocity * delta_t * cos( particles[i].theta );
            y = particles[i].y + velocity * delta_t * sin( particles[i].theta );
            theta = particles[i].theta;
        }
        else{
            x = particles[i].x + (velocity/yaw_rate) * ( sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
            y = particles[i].y + (velocity/yaw_rate) * ( cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t));
            theta = particles[i].theta + yaw_rate * delta_t;
        }
        //create normal distribution
        normal_distribution<double> dist_x(x, std_pos[0]);
        normal_distribution<double> dist_y(y, std_pos[1]);
        normal_distribution<double> dist_theta(theta, std_pos[2]);
        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // TODO: Find the predicted measurement that is closest to each observed measurement and assign the
    //   observed measurement to this particular landmark.
    // NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
    //   implement this method and use it as a helper during the updateWeights phase.

    for (int i = 0; i < observations.size(); ++i)
    {
        // grab current observation
        LandmarkObs current_obs = observations[i];
        // init minimum distancei to maximum
        double min_dist = numeric_limits<double>::max();
        // id of association
        int map_id = -1;

        for (int j = 0; j < predicted.size(); ++j)
        {
            LandmarkObs current_pred = predicted[j];
            double d = dist(current_obs.x, current_obs.y, current_pred.x, current_pred.y);

            if (d < min_dist)
            {
                min_dist = d;
                map_id = current_pred.id;
            }
        }
        // set the observation's id to the nearest predicted landmark's id
        observations[i].id = map_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    //At the end we will recompute the weights

    weights.clear();

    for (int i = 0; i < num_particles; ++i)
    {
        double p_x = particles[i].x;
        double p_y = particles[i].y;
        double p_theta = particles[i].theta;

        vector<LandmarkObs> predictions;

        for (int j = 0; j < map_landmarks.landmark_list.size(); ++j)
        {
            int lm_id = map_landmarks.landmark_list[j].id_i;
            float lm_x = map_landmarks.landmark_list[j].x_f;
            float lm_y = map_landmarks.landmark_list[j].y_f;

            if (dist(lm_x, lm_y, p_x, p_y) <= sensor_range)
            {
                predictions.push_back(LandmarkObs{ lm_id, lm_x, lm_y });
            }
        }

        // Transform observed landmarks to map coordinates
        vector<LandmarkObs> transformed_obs;
        for (int k = 0; k < observations.size(); ++k)
        {
            LandmarkObs tobs;

            tobs.x = p_x + cos(p_theta) * observations[k].x - sin(p_theta) * observations[k].y;
            tobs.y = p_y + sin(p_theta) * observations[k].x + cos(p_theta) * observations[k].y;
            transformed_obs.push_back(tobs);
        }

        // Data Assocication
        dataAssociation(predictions, transformed_obs);

        // Calculate the weights
        particles[i].weight = 1.0;
        for(int l = 0; l < transformed_obs.size(); ++l)
        {
            double obs_x, obs_y, pred_x, pred_y;
            obs_x = transformed_obs[l].x;
            obs_y = transformed_obs[l].y;
            int map_tid = transformed_obs[l].id;

            for (int m = 0; m < predictions.size(); ++m)
            {
                if (predictions[m].id == map_tid)
                {
                    pred_x = predictions[m].x;
                    pred_y = predictions[m].y;
                    break;
                }
            }

            double dx = obs_x - pred_x;
            double dy = obs_y - pred_y;
            double sx = std_landmark[0];
            double sy = std_landmark[1];
            double obs_w = 1 / (2*M_PI*sx*sy) * exp(-0.5*(pow(dx/sx,2)+pow(dy/sy,2)));

            particles[i].weight *= obs_w;
        }
        weights.push_back(particles[i].weight);
    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    discrete_distribution<int> dist(weights.begin(), weights.end());

    vector<Particle> resampled_particles;

    for(int i = 0; i < num_particles; i++)
    {
        resampled_particles.push_back(particles[dist(gen)]);
    }

    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
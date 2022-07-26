#ifndef HUNGARIAN
#define HUNGARIAN

#include <iostream>
#include <vector>
#include <utility>
#include <limits>
#include <cassert>

template<typename T> class Hungarian
{
private:
    const int U, V;
    std::vector<std::vector<int> > graph;
    std::vector<T> dual;
    std::vector<int> alloc, rev_alloc, prev;
    const std::vector<std::vector<T> >& cost;
    int matching_size;
    T diff(const int i, const int j);
    void init_feasible_dual();
    void construct_graph();
    bool find_augmenting_path(const int cur, const int prv, int& pos);
    void update_dual(const T delta);
    void maximum_matching(bool initial=false);
    int dfs(const int cur, const int prv, std::vector<int>& new_ver);
    int increase_matching(const std::vector<std::pair<int, int> >& vec, std::vector<int>& new_ver);
    void hint_increment(int cur);
public:
    Hungarian(const std::vector<std::vector<T> >& _cost);
    std::pair<T, std::vector<int> > solve();
};

#endif

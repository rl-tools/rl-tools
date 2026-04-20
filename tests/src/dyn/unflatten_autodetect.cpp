#include <rl_tools/operations/cpu.h>
#include <rl_tools/dyn/model.h>
#include <rl_tools/dyn/operations_generic.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;

#include <gtest/gtest.h>

// Contract: dyn::Unflatten needs its (height, width, channels) populated — typically
// loaded from the checkpoint's saved attributes. If any are zero the layer has no way
// to disambiguate the flat input (spatial = h*w is underdetermined), so propagate_shapes
// must fail explicitly by setting output_size = 0 on the Unflatten and propagating
// that up through the enclosing composite. An earlier implementation attempted a
// sqrt(spatial) heuristic which only worked for square layouts — that has been removed.

namespace {
    struct UnflattenConvChain {
        rlt::dyn::Layer<TI> seq;
        rlt::dyn::Layer<TI> children_arr[2];
        rlt::dyn::layers::Unflatten<TI> unflatten_data;
        rlt::dyn::layers::Conv2d<TI> conv_data;

        void build(TI input_channels){
            unflatten_data = {};
            unflatten_data.height = 0;
            unflatten_data.width = 0;
            unflatten_data.channels = 0;

            conv_data = {};
            conv_data.input_channels = input_channels;
            conv_data.output_channels = 8;
            conv_data.kernel_height = 3;
            conv_data.kernel_width = 3;
            conv_data.stride_h = 1;
            conv_data.stride_w = 1;
            conv_data.padding_h = 0;
            conv_data.padding_w = 0;

            children_arr[0] = {};
            children_arr[0].type = rlt::dyn::LayerType::UNFLATTEN;
            children_arr[0].data = &unflatten_data;
            children_arr[0].num_children = 0;

            children_arr[1] = {};
            children_arr[1].type = rlt::dyn::LayerType::CONV2D;
            children_arr[1].data = &conv_data;
            children_arr[1].num_children = 0;

            seq = {};
            seq.type = rlt::dyn::LayerType::SEQUENTIAL;
            seq.num_children = 2;
            seq.children = children_arr;
        }
    };
}

// With a well-populated Unflatten the chain propagates cleanly regardless of square vs non-square.
TEST(TEST_DYN_UNFLATTEN_AUTODETECT, PopulatedDimsPropagateNonSquare){
    UnflattenConvChain m;
    m.build(/*input_channels=*/3);
    m.unflatten_data.height = 50;
    m.unflatten_data.width = 80;
    m.unflatten_data.channels = 3;
    TI shape[] = {(TI)1, (TI)(50*80*3)};
    rlt::dyn::propagate_shapes(m.seq, shape, (TI)2);
    EXPECT_GT(m.seq.output_size, (TI)0);
    EXPECT_GT(m.children_arr[0].output_size, (TI)0);
}

// Missing dims → Unflatten must signal failure explicitly. This used to "look like success"
// for square inputs because of the sqrt heuristic; it now fails uniformly.
TEST(TEST_DYN_UNFLATTEN_AUTODETECT, MissingDimsFailLoudlySquare){
    UnflattenConvChain m;
    m.build(/*input_channels=*/3);
    TI shape[] = {(TI)1, (TI)(64*64*3)};
    rlt::dyn::propagate_shapes(m.seq, shape, (TI)2);
    EXPECT_EQ(m.children_arr[0].output_size, (TI)0);
    EXPECT_EQ(m.seq.output_size, (TI)0);
}

TEST(TEST_DYN_UNFLATTEN_AUTODETECT, MissingDimsFailLoudlyNonSquare){
    UnflattenConvChain m;
    m.build(/*input_channels=*/3);
    TI shape[] = {(TI)1, (TI)(50*80*3)};
    rlt::dyn::propagate_shapes(m.seq, shape, (TI)2);
    EXPECT_EQ(m.children_arr[0].output_size, (TI)0);
    EXPECT_EQ(m.seq.output_size, (TI)0);
}

TEST(TEST_DYN_UNFLATTEN_AUTODETECT, MissingDimsFailLoudlyTall){
    UnflattenConvChain m;
    m.build(/*input_channels=*/3);
    TI shape[] = {(TI)1, (TI)(96*60*3)};
    rlt::dyn::propagate_shapes(m.seq, shape, (TI)2);
    EXPECT_EQ(m.children_arr[0].output_size, (TI)0);
    EXPECT_EQ(m.seq.output_size, (TI)0);
}

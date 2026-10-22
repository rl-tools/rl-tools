#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_HSSD_CATALOG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_HSSD_CATALOG_H

#include "../catalog.h"
#include "hssd.h"
#include <cstddef>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::datasets::hssd {
    // scene_splits.yaml of hssd-hab 4369cb9876214c7fbebcf552eb532380e4d287e4, in file order; GLBs from procthor2glb --hssd --normalize (conversion::VERSION 7)
    struct Train125 {
        using LOADER = GLB;
        using Corpus = LOADER::Corpus;
        static constexpr const char* SEGMENTATION_TAXONOMY = LOADER::SEGMENTATION_TAXONOMY;
        static constexpr const char* NAME = "hssd_train_125";
        static constexpr const char* SOURCE = "hssd/hssd-hab";
        static constexpr const char* SPLIT = "train";
        static constexpr const char* REPRESENTATION = "glb";
        inline static constexpr SceneReference SCENES[] = {
            {"103997478_171030525", "conta:076baff95e022dd0f74e6db84c51ccff6eb949aa"},
            {"108736611_177263226", "conta:9ef3a483bc19d22a143f6552e1cced3ef2bbfe04"},
            {"105515175_173104107", "conta:97a03a31118a79fccee8deb079f9db18e113ad5a"},
            {"107734080_175999881", "conta:f05f33e3106b3659c0541477bf29fc2beff4aead"},
            {"103997562_171030642", "conta:efc1e67bae29db439780c92cae1920edde175193"},
            {"108294624_176710203", "conta:3cbc8abdbfecc040e23904b6f32f6a82ef055af9"},
            {"108294816_176710461", "conta:cd24d7579f8796d4bb871f304cfd35e6a42fc667"},
            {"108736722_177263382", "conta:ce09213b4a5228febda5a6a31d6e2607628f6603"},
            {"108294465_176709960", "conta:534eae89f4ca1bb92bfba1aa6bb3a2fddbd4461e"},
            {"106879044_174887172", "conta:d83b942a64c0cc49d5384fe8dab2c5e4d65bcd14"},
            {"102815859", "conta:919fc90b4b69d81cf026b9435f00a23e6b1574fa"},
            {"106366353_174226695", "conta:dd750e5c2f27adbf9ba7de1ed19e245e4633aad1"},
            {"103997718_171030855", "conta:69adb67d7d2780988185cc196bf28ef025312f6c"},
            {"105515286_173104287", "conta:94a9c1c0e1e7628596a46c00d55043a710cb4953"},
            {"103997940_171031257", "conta:88801dd4e3fe14fd2469146516fb7b45c9b69eb2"},
            {"102816216", "conta:d3011e5d204e68cf459751f3088b7f9a139476d1"},
            {"108736872_177263607", "conta:4cba15fa18331de6bdd14dc09e4a46039f543a7d"},
            {"103997478_171030528", "conta:e5355a20147c3326673e060976297a9673c4d43a"},
            {"107734119_175999938", "conta:a1fc0564c9cb54586e0ee44e3cd1a3ebc314668e"},
            {"106366323_174226647", "conta:808680c4fffe9190319c160c7856d9979a2d8898"},
            {"105515505_173104584", "conta:122d4c6513c783e7a47489bd63ddf6c263e067b7"},
            {"108294846_176710506", "conta:c32d1ffa8f8225988a44c040a877937bd617a469"},
            {"106366134_174226362", "conta:0af606fc3c7ef5c69578e2bcfab3b73cc7134a50"},
            {"102816786", "conta:b2ba016fbcdfc251f09287b639d14ba384622301"},
            {"104348253_171513237", "conta:7cd68e2113827c019eedc742a254de9166b8493a"},
            {"105515541_173104641", "conta:9fe07b860a42d83ee91a3486ebd9c388dce32608"},
            {"102816627", "conta:c6c796469a5c937cb52972491b58ea13e7dc46bf"},
            {"104862660_172226844", "conta:261ab3f53d23e5785044eb46453a3ee6d9098b4e"},
            {"108294684_176710278", "conta:e50a8ce6e4d16f408dd1f25ba9cd46cf781666c6"},
            {"106366263_174226557", "conta:feca0e59b940b2cd08c3f33fb80d629f3c342342"},
            {"104348202_171513150", "conta:56a36a72657aa556496dd2a5ac71cc3de2d33d35"},
            {"103997403_171030405", "conta:45d94efbcd9af67e6c40fdf31a2056580c6f405a"},
            {"102816066", "conta:cedec963bf71c3489e550daed10883b4d1989861"},
            {"106366302_174226617", "conta:7aed56b6bb0f22fe089caef16aae7a2278f2fd3d"},
            {"104862639_172226823", "conta:139ad5ef084ee00e5e7ae9ae8071bef9593a010a"},
            {"105515211_173104179", "conta:94b72f8f242317700605440c85f7e606877d1507"},
            {"106878945_174887058", "conta:214e9dca7c32fbbbc2ab8efd9eafe6c7d4fc3bac"},
            {"105515403_173104449", "conta:d094b17f4303e59d436bf186323d651b8db6e94a"},
            {"102344250", "conta:7d10b82dd60cf0dae4c6176862b3163a0eb30440"},
            {"105515151_173104068", "conta:a5b77880e9deee1b26f07a6a650e54f51f85346e"},
            {"104348463_171513588", "conta:55ce378b7428c97cc74e5aae03bf0f260a9a6aab"},
            {"103997613_171030702", "conta:a2de7b21b0826fe1beffc7b92f742a71cdf6c8da"},
            {"104862621_172226772", "conta:223184d79320a956e492bf865e3cb2be379b4064"},
            {"102344457", "conta:98639703a95bb8259ec16b3c241844dfff84883c"},
            {"105515160_173104077", "conta:0ea209c62b4dcb2f5996731d6d5303a0debddca3"},
            {"108294798_176710428", "conta:d4f6b381e6c11bdb3d3f02852081ba207280c8af"},
            {"104862558_172226664", "conta:b7cd305bc8719b6fd0b342b7fccbf65b7f18a777"},
            {"105515490_173104566", "conta:755278b42ba4beec2601f2b187dcc38d50170b97"},
            {"108294783_176710410", "conta:0b93cce1355232fcd6f529053a1baa99671c425c"},
            {"102344022", "conta:d8453efac74c9ecdf8c14d036dd9eb5604e7758b"},
            {"103997970_171031287", "conta:f2d4c2610d373ddd15d93cd9c679537a31906f3e"},
            {"107734119_175999935", "conta:142d78a66adce7142df3c34f7cb862736a986634"},
            {"106879023_174887148", "conta:91f6278b553bcd02d61e21f373d4c473764d0de4"},
            {"106366434_174226881", "conta:af71ba542ee7f9df4f89a1e6ef14550b2b79f695"},
            {"104348082_171512994", "conta:327603846f29053e3756a559ae0f94fe90e455ae"},
            {"108294417_176709879", "conta:7b7db7e00a9b89b7f24da1b0ee2a144f6c3cc52a"},
            {"107733912_175999623", "conta:10e4da9c3df7f00545a8a677200a3077f96b5afa"},
            {"108736635_177263256", "conta:95358b0abb185cc98979a008a13ffd8fc4b2e60e"},
            {"105515430_173104494", "conta:857281dad41e278c8a3547bf0ced842847cb8bda"},
            {"102817200", "conta:6fd10701cb019dc29b276973e9d1735930bc9413"},
            {"104348160_171513093", "conta:61690d486181da10a63111a18d0931d1d977434b"},
            {"103997424_171030444", "conta:f5e80f8b405ebdb9aaeb135bf1aed77ad63ad946"},
            {"107734449_176000403", "conta:4e44846ff143f754be7584097514582d4d63282e"},
            {"107734119_175999932", "conta:b8dd4207a0221dd224f733ebc90627fca2b18011"},
            {"102344328", "conta:a9f10d0c9cce1944cd5785f44be9fecdfbfd6fbc"},
            {"102344193", "conta:92782f282c7975690a7826246fb9c758bd5e55a8"},
            {"104862669_172226853", "conta:aeda0b691bb10eeee98f7c5704eaecfe4d1a8f66"},
            {"108294573_176710113", "conta:38a8bff019905734c23906b2f45e56ca406fe932"},
            {"102344439", "conta:ef56342cce653a46a79570349d45b268fba6532c"},
            {"102344280", "conta:ea1a299e2fb07fd9d0b6132c2775fce29709ab71"},
            {"103997799_171031002", "conta:437b11b2332f046fc32868e2978b6d2e308d74c9"},
            {"106366248_174226527", "conta:f18c926e1d4190c74af388cbb65b1fb1bcfc0476"},
            {"102816051", "conta:6b9debbc4c7be964767f82e528a86d7fdb1a908f"},
            {"106366173_174226431", "conta:e184e95d0a74be18afe0857340aef74fd75f9a06"},
            {"102816615", "conta:a7454de54874bfe493cb7578ab4978844fc18b65"},
            {"108736656_177263304", "conta:3416cca9efd636b379fe759de704cf6cc4e502ed"},
            {"102817053", "conta:1bce530b7b96ba9065dd9472d702b818beb01b50"},
            {"102344049", "conta:4dd51dace956adbd92c1b808ff9a03c8b5cfd3f2"},
            {"106366371_174226743", "conta:b3290560b1e8274da5a8bfe9126c7842049f393f"},
            {"108294870_176710551", "conta:294c35314a8ad495908fbd926197bdf3fdbf1309"},
            {"103997460_171030507", "conta:79e404d38b58c5dddb1a3cd4712fce4b17ac36e2"},
            {"108294558_176710095", "conta:9844be3da02542f33d284fd97c9c2c47c475ac31"},
            {"104862687_172226883", "conta:caffd7e3893b5f4cf412fdcfff5e0fc9ee7c89ce"},
            {"107734017_175999794", "conta:f9aa4211c100ff4706d9c9edc8fab2d43a35d556"},
            {"102344469", "conta:799c1c34da7ce462048f0579f8c748cdf05cbe6d"},
            {"106365897_174225972", "conta:1ad1fc2221e7fa992eb112be78e6d9041625ad0d"},
            {"103997445_171030492", "conta:de4d2947f3e485632b53454ff36537a4ca618a35"},
            {"105515307_173104317", "conta:c5ac991d6da8cc00f3f4463deb1e1430bcffe5c2"},
            {"103997541_171030615", "conta:d9d7dd1aff7f62d3089a464a412ce64e3a72374d"},
            {"105515301_173104305", "conta:e6ac14531a5a39a311afd2184ed3985d2ed50d62"},
            {"104862396_172226349", "conta:b255238fd34e2dee5cdc2b73e99fa92ec6729fab"},
            {"107734188_176000034", "conta:2e1da6673d1a03d018ca0d3c8a191cba220e25dd"},
            {"104348037_171512898", "conta:b7db440c052e87561efc88641727f01ade6e56d0"},
            {"103997919_171031233", "conta:f5a8c4eca729613ae2c6d2a18d19261c3febb7d9"},
            {"107734110_175999914", "conta:2bc7bbeeec0d3acfd4acfd5926706993609c2539"},
            {"108294939_176710668", "conta:37f63e423ac1014bae0e2b17e403d617b42c10fe"},
            {"103997781_171030978", "conta:c2131e5429406b7cd4c4d439bcaa00163900857f"},
            {"104348394_171513453", "conta:fc1415175d73ec91bcdeac302d099509d338137d"},
            {"105515448_173104512", "conta:792d390c940aa5cad3541c6a8bccba5fba23881a"},
            {"107734227_176000091", "conta:cf8f3d10af7129ceb9afb51395ca83695a12b0d4"},
            {"102816729", "conta:a5dc7ac1e5e158afc3cd0e430597d729f0351e62"},
            {"107734146_175999971", "conta:3acf7d3dc814743c570569c6d886826f0d43a0af"},
            {"108294897_176710602", "conta:1770b6f5eb61f0ee84fba1bf89404ae7d9a5047f"},
            {"104348133_171513054", "conta:8b73a920c50cabcf1836f34a61c59a123c6cc69e"},
            {"107734479_176000442", "conta:9a4ac51e7f8a2e967bfc4e20fe0b10f91ce40c19"},
            {"103997643_171030747", "conta:d34d04bdb990b7a722501ff9af8ab5d0b967992c"},
            {"104862534_172226625", "conta:a9b6d1b195521c9ce0689db9d74f1778d5e7346a"},
            {"104348181_171513120", "conta:196d65670076faefced43b30e7100b293d4e8713"},
            {"105515235_173104215", "conta:e3b22290e3bba32c0d9dc8305c5324c0bc7ad9d1"},
            {"104862573_172226682", "conta:74fa0bbb934e221a11380e343cb0c0ae708990d9"},
            {"108736824_177263559", "conta:bffc052a071b39f3c31bc229bdf858583264c526"},
            {"108736851_177263586", "conta:c2538b2ab2a21ae6e919838e4c5f117fa45b91f1"},
            {"108736737_177263406", "conta:2ca8eb34b0d667c49d8902febddff5304bc15ecc"},
            {"102816009", "conta:f384509af55a7ae06258355edb8e798cb62c7464"},
            {"105515337_173104347", "conta:8b26bf4f74712c7232f5db25336aec99795589ed"},
            {"108294765_176710386", "conta:ca632e3e675777ae7683d394ba11ce4a466ef8ed"},
            {"104862681_172226874", "conta:8cdad2985338541605200284f40d01048739357e"},
            {"104862513_172226580", "conta:5fd5611265f3275316a6e2b1741ca760071e74c8"},
            {"102344403", "conta:b12f4726c17f6221300cdfceee445bde427b698c"},
            {"102344307", "conta:6a6cb3c8b9abe9082e4105514b9ea276c0beaac4"},
            {"103997994_171031320", "conta:6d7fd03d2e83407742d27e606aef7ae4b9f4b4a1"},
            {"102816852", "conta:a5c1df79ad37fcc72fb51ee22727f675be96d5a8"},
            {"108294537_176710050", "conta:8eeb0121ef0ee33fcf0fb201a25d1747850d76a8"},
            {"104348511_171513654", "conta:dfc6b263e043c634f3d580b0801b2423d4fe46af"},
            {"102344115", "conta:d4c582adb5354782fce4e7ef76edd23ee5d90ad4"},
        };
        static constexpr std::size_t NUM_SCENES = sizeof(SCENES) / sizeof(SCENES[0]);
    };

    struct Val43 {
        using LOADER = GLB;
        using Corpus = LOADER::Corpus;
        static constexpr const char* SEGMENTATION_TAXONOMY = LOADER::SEGMENTATION_TAXONOMY;
        static constexpr const char* NAME = "hssd_val_43";
        static constexpr const char* SOURCE = "hssd/hssd-hab";
        static constexpr const char* SPLIT = "val";
        static constexpr const char* REPRESENTATION = "glb";
        inline static constexpr SceneReference SCENES[] = {
            {"107734158_175999998", "conta:e5e59b08d09e1192a2a176d05f8006c5d5c21b5f"},
            {"105515211_173104185", "conta:ac8bfd647e844abae191e07dd0dbf4d4abb9b9be"},
            {"106879080_174887211", "conta:d132fa71f3ba8a0c4535acf78e48bd712b3dbabf"},
            {"103997586_171030666", "conta:440a64fc2d7a6538d214aa530531a62914d5c452"},
            {"106878960_174887073", "conta:96d2705ed8b6dea33362f0732be5273e7a4aa4b3"},
            {"106879005_174887124", "conta:85464ff44373c7123e21ebd1ad5359a249b81b3d"},
            {"106878858_174886965", "conta:434e94a692748e22a7c47f52fd06abef1786c793"},
            {"108736884_177263634", "conta:a20fd71751fa8643da2a992d3491f0f4f01ec734"},
            {"106878915_174887025", "conta:7065cad492d5972fac900a595470890c2b2fb1d9"},
            {"107734176_176000019", "conta:bd650df0500eff57d47bd41e40749c9726e12eea"},
            {"104348328_171513363", "conta:62513b62583c5c0b4fb943e6edab0d7d87c39c2e"},
            {"107734254_176000121", "conta:8aa2df35b31bf5cec55c5a4b8b083bbe4f53e9f5"},
            {"107733960_175999701", "conta:d66436c0f23b45f0d8877f2ec78b8a50cb741eac"},
            {"102816756", "conta:edcd0cd2dd53d02510d7993503ceaccc56dd95b5"},
            {"104348361_171513414", "conta:fba0651ffb341e56224a7a16e215678b1fc46f7d"},
            {"108736689_177263340", "conta:4599939ecfae53156a0d75d6f966edc6ed283de5"},
            {"102816600", "conta:edc3cc01cade00a1780ef2cbe6c2f425ac23ab83"},
            {"106366104_174226329", "conta:7584f24f8c2cf828b7dcc0a2574f4cc5df4341a8"},
            {"108736779_177263484", "conta:b906da0472e42eef019f75d05b2e9574e9b16443"},
            {"104862384_172226319", "conta:e3e783ff4b0d62ee87f74b8b07e4090b90298f36"},
            {"102816036", "conta:aef22d6969d90f612e17de21a875962b9088638f"},
            {"107734287_176000160", "conta:7b70eee6e46e6ae7a21f092335f0787f02caaceb"},
            {"105515379_173104395", "conta:0dea4d4817021301fa56d2db80216f496201aab1"},
            {"106366386_174226770", "conta:41d02e7938d94f0df5f3f245b0800d070820fced"},
            {"102817140", "conta:dd0be90f9ea73b86ce3be5b0513447f19183ed99"},
            {"104348010_171512832", "conta:a81a1a707c0e29b08e8ddb414e41edc5e0872830"},
            {"103997895_171031182", "conta:85b69647e9f17854babfa16d7bee7b38fbb88414"},
            {"106878867_174886977", "conta:21d4e25e04feef63f73ef40e8111f5c8cf2aea4f"},
            {"103997586_171030669", "conta:22af358e205e0e68d5a4e0c56ff66315fb8b0eb1"},
            {"102815835", "conta:1ff084a3eb49979f211bb59bc0e50668535c830a"},
            {"104862417_172226382", "conta:12b9568e84d40c948130cce9b45886f332db7ff4"},
            {"104348028_171512877", "conta:535945e1126ec76cefecce1dfdb946c7dfa166ae"},
            {"104862501_172226556", "conta:f65828205a87ae7a15e31511c0b221ecb57e4391"},
            {"106366104_174226332", "conta:80c0dadef088e60942638fb60c7ccb6991c63b54"},
            {"107734338_176000244", "conta:8eb5910d46c730499266dc4d34fee8fa82564532"},
            {"102343992", "conta:d0ac9de04c5c5663d0fb56313b997e458ecec6a0"},
            {"107734056_175999839", "conta:9333880fa1ddf0202dadff1ce058eba35585288a"},
            {"105515184_173104128", "conta:b47e089da152038103af68b431d655b1c750427a"},
            {"102816114", "conta:880204c4c404dcb4f8d3302e186575c7d3a68a36"},
            {"102344094", "conta:3b05f3ce9ea67a8bf62790fcd7021e640aba4771"},
            {"106366410_174226806", "conta:1140abc68804044c330fd6f865fe2fa2ecff6e87"},
            {"102344529", "conta:d0be4d4856c4ee6a133a560288ec2c24888cccde"},
            {"108736800_177263517", "conta:f5b7ede1dc262109b4e85185803c7ce82b5ac958"},
        };
        static constexpr std::size_t NUM_SCENES = sizeof(SCENES) / sizeof(SCENES[0]);
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif

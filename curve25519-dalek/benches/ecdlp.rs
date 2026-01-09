use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use curve25519_dalek::{
    EdwardsPoint,
    Scalar,
    constants::RISTRETTO_BASEPOINT_POINT as G,
    ecdlp::{self, AffineMontgomeryPoint, DefaultScheduler, ECDLPArguments, ECDLPTables}
};
use rand::{Rng, rng};
use std::{path::Path, time::Duration};

pub fn ecdlp_bench(c: &mut Criterion) {
    if !Path::new("ecdlp_table.bin").exists() {
        let tables = ECDLPTables::generate(26).expect("Failed to generate ECDLP tables");
        tables
            .write_to_file("ecdlp_table.bin")
            .expect("Failed to write ECDLP tables to file");
    }

    let tables = ECDLPTables::load_from_file(26, "ecdlp_table.bin")
        .expect("Failed to load ECDLP tables from file");
    let view = tables.view();

    c.bench_function("fast ecdlp non constant time", |b| {
        let num = 1u64 << 46;
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48).pseudo_constant_time(true),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function("fast ecdlp", |b| {
        let num = rng().random_range(0u64..(1 << 48));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48).pseudo_constant_time(true),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function("par fast ecdlp T=1", |b| {
        let num = rng().random_range(0u64..(1 << 48));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::par_decode::<DefaultScheduler, _>(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48)
                    .pseudo_constant_time(true)
                    .n_threads(1),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function("par fast ecdlp T=2", |b| {
        let num = rng().random_range(0u64..(1 << 48));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::par_decode::<DefaultScheduler, _>(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48)
                    .pseudo_constant_time(true)
                    .n_threads(2),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function("par fast ecdlp T=4", |b| {
        let num = rng().random_range(0u64..(1 << 48));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::par_decode::<DefaultScheduler, _>(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48)
                    .pseudo_constant_time(true)
                    .n_threads(4),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function("par fast ecdlp T=8", |b| {
        let num = rng().random_range(0u64..(1 << 48));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::par_decode::<DefaultScheduler, _>(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48)
                    .pseudo_constant_time(true)
                    .n_threads(8),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function("fast ecdlp find 0", |b| {
        let num: u64 = 0;
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 47)), |b| {
        let num = rng().random_range(0u64..(1 << 47));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(
        &format!("fast ecdlp for number < {} T=2", (1i64 << 47)),
        |b| {
            let num = rng().random_range(0u64..(1 << 47));
            let point = Scalar::from(num) * G;
            b.iter(|| {
                let res = ecdlp::par_decode::<DefaultScheduler, _>(
                    &view,
                    black_box(point),
                    ECDLPArguments::new_with_range(0, 1 << 48).n_threads(1),
                );
                assert_eq!(res, Some(num as i64));
            });
        },
    );

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 44)), |b| {
        let num = rng().random_range(0u64..(1 << 44));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 43)), |b| {
        let num = rng().random_range(0u64..(1 << 43));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 26)), |b| {
        let num = rng().random_range(0u64..(1 << 26));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });

    c.bench_function(&format!("fast ecdlp for number < {}", (1i64 << 27)), |b| {
        let num = rng().random_range(0u64..(1 << 27));
        let point = Scalar::from(num) * G;
        b.iter(|| {
            let res = ecdlp::decode(
                &view,
                black_box(point),
                ECDLPArguments::new_with_range(0, 1 << 48),
            );
            assert_eq!(res, Some(num as i64));
        });
    });
}

fn bench_table_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Table Generation");

    // Configure the benchmark for faster execution
    // Reduce to minimum samples
    group.sample_size(10);
    // Less measurement time
    group.measurement_time(Duration::from_secs(5));

    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);

    for l1 in [13, 18, 21].iter() {
        group.bench_with_input(BenchmarkId::new("Sequential", l1), l1, |b, &l1| {
            b.iter(|| ECDLPTables::generate(l1).expect("Failed to generate ECDLP tables"));
        });

        group.bench_with_input(BenchmarkId::new("Parallel", l1), l1, |b, &l1| {
            b.iter(|| {
                ECDLPTables::generate_par::<DefaultScheduler>(l1, n_threads)
                    .expect("Failed to generate ECDLP tables in parallel")
            });
        });
    }

    group.finish();
}

fn bench_montgomery(c: &mut Criterion) {
    let mut group = c.benchmark_group("Affine Montgomery Point");

    let ed_p1 = EdwardsPoint::mul_base(&Scalar::from(2u64));
    let ed_p2 = EdwardsPoint::mul_base(&Scalar::from(3u64));
    let ed_p3 = EdwardsPoint::mul_base(&Scalar::from(5u64));
    let ed_p4 = EdwardsPoint::mul_base(&Scalar::from(7u64));
    let ed_addend = EdwardsPoint::mul_base(&Scalar::from(11u64));

    let p1 = AffineMontgomeryPoint::from(&ed_p1);
    let p2 = AffineMontgomeryPoint::from(&ed_p2);
    let p3 = AffineMontgomeryPoint::from(&ed_p3);
    let p4 = AffineMontgomeryPoint::from(&ed_p4);
    let addend = AffineMontgomeryPoint::from(&ed_addend);

    group.bench_function("add", |b| {
        b.iter(|| {
            let points = [p1, p2, p3, p4];
            for p1 in points.iter() {
                let _ = black_box(p1).addition_not_ct(black_box(&addend)); 
            }
        });
    });

    group.bench_function("batch add", |b| {
        b.iter(|| {
            let mut points = [p1, p2, p3, p4];
            let _ = AffineMontgomeryPoint::batch_addition_not_ct(black_box(&mut points), black_box(&addend));
        });
    });

    group.finish();
}

criterion_group!(ecdlp, ecdlp_bench);
criterion_group!(tables, bench_table_generation);
criterion_group!(montgomery, bench_montgomery);
criterion_main!(ecdlp, tables, montgomery);
<?php

use Illuminate\Support\Facades\Route;

Route::get('/radio/{slug}', function (string $slug) {
    return view('public-radio', [
        'slug' => $slug,
        'embed' => false,
    ]);
});

Route::get('/embed/radio/{slug}', function (string $slug) {
    return view('public-radio', [
        'slug' => $slug,
        'embed' => true,
    ]);
});

Route::get('/{any?}', function () {
    return view('app');
})->where('any', '.*');
